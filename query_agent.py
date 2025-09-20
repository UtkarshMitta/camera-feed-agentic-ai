"""
LangGraph-based agentic query system for camera feed analysis.
"""
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from .mcp_tools import MCPTools
import json


class QueryState(TypedDict):
    messages: List[dict]
    query: str
    intent: Optional[Dict[str, Any]]
    data_results: Optional[Dict[str, Any]]
    response: Optional[str]
    error: Optional[str]


class QueryAgent:
    def __init__(self, openai_api_key: str, data_dir: str = "."):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.1
        )
        self.mcp_tools = MCPTools(data_dir)
        self.tools = self._create_tools()
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _create_tools(self) -> List:
        @tool
        def get_all_camera_feeds() -> str:
            result = self.mcp_tools.get_all_camera_feeds()
            return json.dumps(result, indent=2)

        @tool
        def filter_by_theater(theater: str) -> str:
            result = self.mcp_tools.filter_by_theater(theater)
            return json.dumps(result, indent=2)

        @tool
        def filter_by_codec(codec: str) -> str:
            result = self.mcp_tools.filter_by_codec(codec)
            return json.dumps(result, indent=2)

        @tool
        def filter_by_resolution(min_width: Optional[int] = None, 
                               min_height: Optional[int] = None,
                               max_width: Optional[int] = None,
                               max_height: Optional[int] = None) -> str:
            result = self.mcp_tools.filter_by_resolution(min_width, min_height, max_width, max_height)
            return json.dumps(result, indent=2)

        @tool
        def filter_by_latency(max_latency: Optional[int] = None,
                             min_latency: Optional[int] = None) -> str:
            result = self.mcp_tools.filter_by_latency(max_latency, min_latency)
            return json.dumps(result, indent=2)

        @tool
        def get_high_quality_feeds() -> str:
            result = self.mcp_tools.get_high_quality_feeds()
            return json.dumps(result, indent=2)

        @tool
        def filter_by_model(model_tag: str) -> str:
            result = self.mcp_tools.filter_by_model(model_tag)
            return json.dumps(result, indent=2)

        @tool
        def filter_by_encryption(encrypted: bool = True) -> str:
            result = self.mcp_tools.filter_by_encryption(encrypted)
            return json.dumps(result, indent=2)

        @tool
        def get_encoder_parameters() -> str:
            result = self.mcp_tools.get_encoder_parameters()
            return json.dumps(result, indent=2)

        @tool
        def get_decoder_parameters() -> str:
            result = self.mcp_tools.get_decoder_parameters()
            return json.dumps(result, indent=2)

        @tool
        def analyze_theater_distribution() -> str:
            result = self.mcp_tools.analyze_theater_distribution()
            return json.dumps(result, indent=2)

        @tool
        def analyze_codec_distribution() -> str:
            result = self.mcp_tools.analyze_codec_distribution()
            return json.dumps(result, indent=2)

        @tool
        def search_feeds(**filters) -> str:
            result = self.mcp_tools.search_feeds(**filters)
            return json.dumps(result, indent=2)

        return [
            get_all_camera_feeds,
            filter_by_theater,
            filter_by_codec,
            filter_by_resolution,
            filter_by_latency,
            get_high_quality_feeds,
            filter_by_model,
            filter_by_encryption,
            get_encoder_parameters,
            get_decoder_parameters,
            analyze_theater_distribution,
            analyze_codec_distribution,
            search_feeds
        ]

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(QueryState)
        workflow.add_node("parse_query", self._parse_query)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("generate_response", self._generate_response)
        workflow.set_entry_point("parse_query")
        workflow.add_edge("parse_query", "execute_tools")
        workflow.add_edge("execute_tools", "generate_response")
        workflow.add_edge("generate_response", END)
        return workflow.compile()

    def _parse_query(self, state: QueryState) -> QueryState:
        query = state["query"]
        prompt = f"""
        Analyze this camera feed query and extract the intent and parameters:
        Query: "{query}"
        Extract:
        1. Intent type (filter, search, analyze, get_info)
        2. Theater (CONUS, PAC, EUR, ME, AFR, ARC) if mentioned
        3. Codec (H264, H265, AV1, VP9, MPEG2) if mentioned
        4. Resolution requirements (4K, 1080p, 720p, etc.)
        5. Quality requirements (best clarity, high quality, etc.)
        6. Latency requirements (low latency, real-time, etc.)
        7. Other filters (encrypted, civilian safe, etc.)
        Return as JSON format:
        {{
            "intent": "filter|search|analyze|get_info",
            "theater": "theater_code_or_null",
            "codec": "codec_or_null",
            "resolution": "resolution_requirement_or_null",
            "quality": "quality_requirement_or_null",
            "latency": "latency_requirement_or_null",
            "other_filters": {{"key": "value"}}
        }}
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            intent_data = json.loads(content)
            state["intent"] = intent_data
        except Exception as e:
            query_lower = query.lower()
            intent_data = {
                "intent": "search",
                "theater": None,
                "codec": None,
                "resolution": None,
                "quality": None,
                "latency": None,
                "other_filters": {}
            }
            if "pacific" in query_lower or "pac" in query_lower:
                intent_data["theater"] = "PAC"
            if "europe" in query_lower or "eur" in query_lower:
                intent_data["theater"] = "EUR"
            if "h265" in query_lower:
                intent_data["codec"] = "H265"
            if "4k" in query_lower:
                intent_data["resolution"] = "4K"
            if "best" in query_lower or "quality" in query_lower:
                intent_data["quality"] = "high"
            state["intent"] = intent_data
        return state

    def _execute_tools(self, state: QueryState) -> QueryState:
        intent = state.get("intent", {})
        if intent.get("intent") == "error":
            state["data_results"] = {"error": intent.get("error", "Unknown error")}
            return state
        try:
            tool_calls = []
            if intent.get("intent") in ["filter", "search", "analyze"]:
                if intent.get("theater"):
                    tool_calls.append(("filter_by_theater", {"theater": intent["theater"]}))
                elif intent.get("codec"):
                    tool_calls.append(("filter_by_codec", {"codec": intent["codec"]}))
                elif intent.get("quality") == "high" or "best" in intent.get("quality", "").lower():
                    tool_calls.append(("get_high_quality_feeds", {}))
                else:
                    tool_calls.append(("get_all_camera_feeds", {}))
            results = {}
            for tool_name, params in tool_calls:
                tool_func = next(tool for tool in self.tools if tool.name == tool_name)
                result = tool_func.invoke(params)
                results[tool_name] = json.loads(result)
            state["data_results"] = results
        except Exception as e:
            state["error"] = f"Error executing tools: {str(e)}"
            state["data_results"] = {"error": str(e)}
        return state

    def _generate_response(self, state: QueryState) -> QueryState:
        query = state["query"]
        intent = state.get("intent", {})
        data_results = state.get("data_results", {})
        if state.get("error"):
            state["response"] = f"I encountered an error: {state['error']}"
            return state
        prompt = f"""
        Based on the user's query and the data results, generate a helpful response.
        User Query: "{query}"
        Intent: {json.dumps(intent, indent=2)}
        Data Results: {json.dumps(data_results, indent=2)}
        Guidelines:
        1. Provide a clear, natural language response
        2. Include specific camera feed IDs when relevant
        3. Explain technical terms in user-friendly language
        4. If filtering results, show the count and key details
        5. If asking about quality, explain what makes feeds high quality
        6. Be conversational and helpful
        Response:
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["response"] = response.content
        except Exception as e:
            state["response"] = f"I encountered an error generating the response: {str(e)}"
        return state

    def query(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "messages": [],
            "query": query,
            "intent": None,
            "data_results": None,
            "response": None,
            "error": None
        }
        result = self.graph.invoke(initial_state)
        return {
            "query": query,
            "response": result["response"],
            "intent": result.get("intent"),
            "data_results": result.get("data_results"),
            "error": result.get("error")
        }
