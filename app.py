"""
Streamlit UI for the Agentic Camera Feed Query System.
"""
import streamlit as st
import os
from dotenv import load_dotenv
from src.query_agent import QueryAgent
import json
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Camera Feed Query System",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .response-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agent():
    """Initialize the query agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set your OPENAI_API_KEY in the .env file")
        st.stop()
    
    if "agent" not in st.session_state:
        with st.spinner("Initializing query system..."):
            st.session_state.agent = QueryAgent(api_key)
    return st.session_state.agent

def display_data_overview():
    """Display overview of the camera feed data."""
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Camera Feeds", "100")
    
    with col2:
        st.metric("Geographic Theaters", "6")
    
    with col3:
        st.metric("Video Codecs", "5")
    
    with col4:
        st.metric("Resolution Range", "480p - 4K")

def display_sample_queries():
    """Display sample queries for users."""
    st.subheader("💡 Sample Queries")
    
    sample_queries = [
        "What are the camera IDs capturing the Pacific area with the best clarity?",
        "Show me all 4K cameras in the Pacific region",
        "Which cameras have the lowest latency for real-time monitoring?",
        "Find encrypted feeds with H265 codec and high frame rates",
        "What's the best quality camera for surveillance in Europe?",
        "Show me all civilian-safe cameras in the Middle East",
        "Which cameras use the Viper-VL analytics model?",
        "Find all cameras with latency under 200ms"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        if st.button(f"{i}. {query}", key=f"sample_{i}", use_container_width=True):
            st.session_state.query_input = query

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">📹 Agentic Camera Feed Query System</h1>', unsafe_allow_html=True)
    
    # Initialize agent
    agent = initialize_agent()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Controls")
        
        # Data overview
        display_data_overview()
        
        st.divider()
        
        # Sample queries
        display_sample_queries()
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.info("""
        This system uses LangGraph for agentic workflows and MCP tools for data operations.
        
        **Features:**
        - Natural language query processing
        - Camera feed filtering and analysis
        - Quality metrics calculation
        - Geographic and technical filtering
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Query Interface")
        
        # Query input
        query_input = st.text_area(
            "Enter your query about camera feeds:",
            value=st.session_state.get("query_input", ""),
            height=100,
            placeholder="e.g., What are the camera IDs capturing the Pacific area with the best clarity?"
        )
        
        # Query buttons
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🚀 Query", type="primary", use_container_width=True):
                if query_input.strip():
                    with st.spinner("Processing query..."):
                        result = agent.query(query_input)
                        st.session_state.last_result = result
                else:
                    st.warning("Please enter a query")
        
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.query_input = ""
                st.session_state.last_result = None
                st.rerun()
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        # Get some quick statistics
        try:
            mcp_tools = agent.mcp_tools
            
            # Theater distribution
            theater_stats = mcp_tools.analyze_theater_distribution()
            st.metric("Theater Distribution", f"{len(theater_stats['theater_distribution'])} regions")
            
            # Codec distribution
            codec_stats = mcp_tools.analyze_codec_distribution()
            st.metric("Codec Types", f"{len(codec_stats['codec_distribution'])} formats")
            
            # High quality feeds
            quality_feeds = mcp_tools.get_high_quality_feeds()
            st.metric("High Quality Feeds", f"{quality_feeds['count']} feeds")
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    # Display results
    if "last_result" in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result
        
        st.divider()
        st.subheader("📋 Query Results")
        
        # Response
        st.markdown('<div class="response-box">', unsafe_allow_html=True)
        st.write(result["response"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results (expandable)
        with st.expander("🔍 Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Intent Analysis")
                if result.get("intent"):
                    st.json(result["intent"])
                else:
                    st.info("No intent analysis available")
            
            with col2:
                st.subheader("Data Results")
                if result.get("data_results"):
                    # Try to display as table if it's feed data
                    data_results = result["data_results"]
                    for tool_name, data in data_results.items():
                        if isinstance(data, dict) and "feeds" in data:
                            st.subheader(f"Results from {tool_name}")
                            if data["feeds"]:
                                df = pd.DataFrame(data["feeds"])
                                st.dataframe(df, use_container_width=True)
                            else:
                                st.info("No feeds found matching the criteria")
                        else:
                            st.subheader(f"Results from {tool_name}")
                            st.json(data)
                else:
                    st.info("No data results available")
        
        # Error handling
        if result.get("error"):
            st.error(f"Error: {result['error']}")

if __name__ == "__main__":
    main()
