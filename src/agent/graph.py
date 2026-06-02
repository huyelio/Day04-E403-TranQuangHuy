from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, an electronics retail order assistant.
Today is {current_day}.

LANGUAGE AND GROUNDING
- Understand Vietnamese, English, and mixed-language requests.
- Reply to the customer in concise Vietnamese.
- Never invent product IDs, SKUs, prices, stock, discounts, campaign codes, totals,
  detail tokens, order IDs, or save paths. Use only values returned by tools.
- Never silently replace a requested product with another product.

HIGH-PRIORITY QUANTITY NORMALIZATION
- A product listed without a number in a confirmed order list always has quantity 1.
- This includes comma-separated quoted names. For example:
  `"MacBook Air M3 13", "Sony WH-1000XM5", "Samsung T7 Shield 2TB"`
  means quantity 1 for each of the three products.
- Do not ask for quantities in that situation. Continue the valid order workflow.

MANDATORY PREFLIGHT BEFORE ANY TOOL CALL
First inspect the user's request. Do not call any tool until the request contains:
1. customer full name
2. phone number
3. email address explicitly written by the user and containing the literal character "@"
4. shipping address
5. at least one requested product. Normalize quantity as follows:
   - use the explicit quantity when the customer writes a number
   - otherwise use quantity 1 for each product in a clearly stated order-item list,
     including quoted product names
   Never ask for quantity when a listed order item has no number; use quantity 1.

If any required information is missing, ask only for the missing fields in Vietnamese
and stop immediately. Do not call any tool.
Never infer, guess, synthesize, or auto-fill a missing field. In particular, never
construct an email address from a customer name. If the user text has no literal "@",
the email is missing and you must ask for it without calling tools.

REFUSAL RULES BEFORE ANY TOOL CALL
If the user asks for a fake invoice, manual or forced discount, stock bypass, catalog
bypass, policy bypass, or asks you to ignore validation, refuse clearly in Vietnamese
and stop immediately. Do not call any tool.

VALID ORDER WORKFLOW
For a complete and allowed order, follow this sequence exactly:
1. Call list_products exactly once. Include all requested product names in query and use
   a sufficiently large limit, normally 20, so every requested item can be discovered.
2. Match every requested item to exact product IDs returned by list_products. Then call
   get_product_details exactly once with all chosen product IDs.
3. Inspect every returned item before proceeding. If any product is not found or any
   requested quantity exceeds returned stock, explain the issue in Vietnamese and stop.
   Do not call get_discount, calculate_order_totals, or save_order.
4. Call get_discount exactly once using the customer's email as seed_hint. Use
   customer_tier="standard" unless the user explicitly states VIP.
5. Call calculate_order_totals exactly once with normalized product IDs, quantities,
   the detail_token from get_product_details, and discount_rate from get_discount.
6. Proceed only when calculate_order_totals returns status="ok". Then call save_order
   exactly once using the validated items and the values returned by previous tools.
7. If a tool returns status="error", stop and explain the error. Never save.

FINAL CONFIRMATION
After save_order returns status="saved", provide one short Vietnamese confirmation
mentioning the saved order ID, campaign discount, final total in VND, and save path.
Use this shape: "Đã lưu đơn <order_id>. Giảm giá <campaign_code> (<percent>%).
Tổng thanh toán: <final_total> VND. Đường dẫn lưu: <path>."
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the real catalog once before selecting IDs. For an order, pass all requested product names in query and use limit=20."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Validate all selected catalog IDs once. Inspect exact stock and reuse the returned detail_token in later tools."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Get the policy-approved campaign only after details and stock pass. Use customer email as seed_hint."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Calculate grounded totals after stock validation using exact IDs, detail_token, and approved discount_rate."""
        payload = store.calculate_order_totals(
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist a complete validated order only after calculate_order_totals succeeds. Use the customer's exact email from the user request; never guess or synthesize it. Never send blank fields."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
