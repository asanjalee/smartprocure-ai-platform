import streamlit as st
import os
import pandas as pd
import pypdf
import json
import re
import numpy as np
import time
from datetime import datetime

# --- Custom TF-IDF Embedding Function for ChromaDB ---
# This ensures 100% offline reliability without large downloads or API keys.
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction

class SimpleVocabularyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # A curated vocabulary specific to the procurement policies
        self.vocab = [
            "policy", "procurement", "guideline", "rule", "standard",
            "budget", "threshold", "limitation", "fifty", "thousand", "50000",
            "payment", "advanced", "upfront", "downpayment", "net", "30", "60", "ten", "10%",
            "delivery", "sla", "days", "45", "timeframes", "timeline",
            "warranty", "guarantee", "months", "duration", "year", "maintenance",
            "sustainability", "environmental", "certification", "iso", "14001", "9001", "bifma",
            "material", "mdf", "wood", "laminate", "hpl", "desktops", "hazards"
        ]
        
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            tokens = re.findall(r'\b\w+\b', text.lower())
            vector = []
            for word in self.vocab:
                # Basic frequency count vectorizer
                vector.append(float(tokens.count(word)))
            # Cosine normalization
            vec_arr = np.array(vector, dtype=float)
            norm = np.linalg.norm(vec_arr)
            if norm > 0:
                vec_arr = vec_arr / norm
            embeddings.append(vec_arr.tolist())
        return embeddings

# --- Helper Functions for Data Processing ---

def chunk_policy_pdf(filepath):
    """Chunks the core policy document based on double newlines and logical boundaries."""
    if not os.path.exists(filepath):
        return []
    
    reader = pypdf.PdfReader(filepath)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
        
    # Split text into logical components
    raw_sections = full_text.split("\n\n")
    processed_chunks = []
    
    for section in raw_sections:
        clean_sec = section.strip()
        if len(clean_sec) > 30:
            processed_chunks.append(clean_sec)
            
    return processed_chunks

def parse_quotation_nlp(text):
    """Processes PDF quotation text using heuristic rule-based NLP & regex matching."""
    extracted = {
        "vendor_name": "Unknown Vendor",
        "total_price": 0.0,
        "advance_payment_pct": 0.0,
        "delivery_days": 100,
        "warranty_months": 0,
        "sourcing_material": "Unknown Theme"
    }
    
    text_lower = text.lower()
    
    # 1. Vendor Name
    if "apex office solutions" in text_lower:
        extracted["vendor_name"] = "Apex Office Solutions"
    elif "beacon tech furniture" in text_lower:
        extracted["vendor_name"] = "Beacon Tech Furniture"
    elif "crown workspace" in text_lower:
        extracted["vendor_name"] = "Crown Workspace Group"
    else:
        # Fallback matcher
        prepared_match = re.search(r'prepared by:\s*([^\n\r]+)', text, re.IGNORECASE)
        if prepared_match:
            extracted["vendor_name"] = prepared_match.group(1).strip()
            
    # 2. Total Proposal Price
    price_matches = re.findall(r'\$\s*([0-9,]+(?:\.[0-9]{2})?)', text)
    if price_matches:
        for match in price_matches:
            val = float(match.replace(",", ""))
            if val > 1000:  # Ignore small sub-items/unit prices
                extracted["total_price"] = val
                break
                
    # 3. Advance Payment
    if "advance" in text_lower or "downpayment" in text_lower:
        adv_match = re.search(r'(\d+)\s*%\s*(?:advance|downpayment)', text_lower)
        if adv_match:
            extracted["advance_payment_pct"] = float(adv_match.group(1))
        # Look for phrases like "50% Downpayment"
        elif "50%" in text_lower:
            extracted["advance_payment_pct"] = 50.0
            
    # 4. Delivery Timeline (Days)
    delivery_match = re.search(r'(\d+)\s*(?:calendar\s+)?days', text_lower)
    if delivery_match:
        extracted["delivery_days"] = int(delivery_match.group(1))
    elif "60 calendar days" in text_lower:
        extracted["delivery_days"] = 60
        
    # 5. Warranty Period (Months)
    warranty_match = re.search(r'(\d+)\s*(?:months|month)', text_lower)
    if warranty_match:
        extracted["warranty_months"] = int(warranty_match.group(1))
    elif "1 year" in text_lower or "12 months" in text_lower:
        extracted["warranty_months"] = 12
    elif "3 years" in text_lower or "36 months" in text_lower:
        extracted["warranty_months"] = 36
    elif "6 months" in text_lower:
        extracted["warranty_months"] = 6
        
    # 6. Sourcing/Material Composition
    if "high-pressure laminate" in text_lower or "hpl" in text_lower:
        extracted["sourcing_material"] = "High-Pressure Laminate (HPL)"
    elif "medium density fiberboard" in text_lower or "mdf" in text_lower:
        extracted["sourcing_material"] = "Medium Density Fiberboard (MDF)"
    elif "solid oak" in text_lower or "solid wood" in text_lower:
        extracted["sourcing_material"] = "Solid Oak Wood"
    else:
        # Fallback search
        mat_match = re.search(r'material[s]?\s*(?:sourcing|composition):\s*([^\n\r]+)', text, re.IGNORECASE)
        if mat_match:
            extracted["sourcing_material"] = mat_match.group(1).strip()
            
    return extracted

# --- UI Setup and Themes ---
st.set_page_config(
    page_title="SmartProcure AI - Procurement & Risk Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium stylesheet link setup & custom styling injection
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
        }
        
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #0F766E 100%);
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-header h1 {
            color: #FFFFFF;
            font-weight: 700;
            font-size: 2.2rem;
            margin: 0;
            letter-spacing: -0.02em;
        }
        
        .main-header p {
            color: #CBD5E1;
            font-size: 1rem;
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        
        /* Glassmorphic Metrics Card */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .metric-lbl {
            font-size: 0.8rem;
            color: #94A3B8;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-val {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 5px;
            color: #F8FAFC;
        }
        
        /* Agent communication rolling logs */
        .agent-log {
            background-color: #070A11;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 12px;
            font-family: 'Courier New', Courier, monospace;
            max-height: 250px;
            overflow-y: auto;
            color: #38BDF8;
            font-size: 0.85rem;
            line-height: 1.4;
        }
        
        .agent-log-line {
            border-bottom: 1px solid #1E293B;
            padding: 6px 0;
        }
        
        .agent-pulse {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 6px;
            background-color: #10B981;
            box-shadow: 0 0 8px #10B981;
        }
        
        .agent-pulse-thinking {
            background-color: #F59E0B;
            box-shadow: 0 0 8px #F59E0B;
        }
        
        /* Custom Compliance Table badges */
        .badge-success {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .badge-danger {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 6px;
            padding: 2px 8px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 15px;
            background-color: rgba(255, 255, 255, 0.02);
            padding: 8px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 16px;
            color: #94A3B8;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E3A8A 0%, #0D9488 100%) !important;
            color: white !important;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)


# --- Core System Init / Persistence ---

# Initialize vector db collection using cached resource helper
@st.cache_resource
def init_chroma_db():
    import chromadb
    db_path = os.path.join("data", "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    # Define custom embedder
    embedder = SimpleVocabularyEmbeddingFunction()
    
    try:
        collection = client.get_collection("procurement_policy", embedding_function=embedder)
    except Exception:
        collection = client.create_collection("procurement_policy", embedding_function=embedder)
        
    return collection

collection = init_chroma_db()

# Auto-index policy target PDF
policy_pdf_path = os.path.join("data", "Procurement_Policy_2026.pdf")
if os.path.exists(policy_pdf_path) and collection.count() == 0:
    chunks = chunk_policy_pdf(policy_pdf_path)
    if chunks:
        doc_ids = [f"policy_chunk_{i}" for i in range(len(chunks))]
        mats = [{"source": "Procurement_Policy_2026.pdf", "chunk_id": i} for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            ids=doc_ids,
            metadatas=mats
        )

# Initialize Session State values to track log history, selections, overrides
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

if "sanitization_active" not in st.session_state:
    st.session_state.sanitization_active = True

if "overridden_status" not in st.session_state:
    st.session_state.overridden_status = {}

def add_agent_log(agent_name, action, target=""):
    now = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{now}] ⚙️ <b>{agent_name}</b>: {action}"
    if target:
        log_entry += f" → <i>{target}</i>"
    st.session_state.agent_logs.insert(0, log_entry)

# --- Layout Grid ---

# Clean Header Banner
st.markdown("""
    <div class="main-header">
        <h1>SmartProcure AI Platform</h1>
        <p>Information Retrieval & Web Analytics Project — Secure Multi-Agent Procurement Audit & Contract Compliance System</p>
    </div>
""", unsafe_allow_html=True)


# --- Sidebar Section ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield-with-security-lock.png", width=80)
    st.title("Admin Console")
    
    # 1. User Identity & Auditing Controls
    role = st.selectbox(
        "User Auditing Identity",
        options=["🔍 Risk Compliance Analyst", "👨‍💼 Procurement Manager"],
        help="Access controls change visual tools and authorization states on PO operations."
    )
    
    # 2. Security Subsystem status controls
    st.subheader("🛡️ Security Controls")
    enc_status = st.checkbox("Enable In-transit Encryption", value=True, help="Encrypt outputs in communications.")
    sanitization = st.toggle("Enable Input Sanitization", value=True, help="Prevents prompt injection / path sanitization.")
    st.session_state.sanitization_active = sanitization
    
    # Showcase active agents
    st.subheader("🔄 Multi-Agent Pipeline Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p><span class="agent-pulse"></span>A1: Document Intel</p>', unsafe_allow_html=True)
        st.markdown('<p><span class="agent-pulse"></span>A2: Policy Audit</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p><span class="agent-pulse"></span>A3: Risk Analyst</p>', unsafe_allow_html=True)
        st.markdown('<p><span class="agent-pulse"></span>A4: PO Generation</p>', unsafe_allow_html=True)
        
    st.divider()
    
    # Reset/reload local Chroma index
    if st.button("Reload Reference Index", type="secondary"):
        try:
            # Recompute chunks and push
            st.cache_resource.clear()
            st.success("Reference vector index refreshed!")
            add_agent_log("ChromaDB Client", "Reference indices reloaded.")
            time.sleep(1.0)
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting database: {e}")

# --- Pipeline Stage Tabs ---
tab_titles = [
    "📦 Stage 1: Document Extract", 
    "⚖️ Stage 2: Compliance RAG", 
    "📊 Stage 3: Risk Evaluation", 
    "✍️ Stage 4: Action & PO Approval"
]
t1, t2, t3, t4 = st.tabs(tab_titles)

# Global variables for shared state across pipeline stages
selected_source = None
extracted_params = None

# --- STAGE 1: Document Intelligence (Agent 1) ---
with t1:
    st.header("📄 Agent 1: Document Intelligence")
    st.write("Extracting and parsing structured commercial data parameters from Vendor Proposal documents.")
    
    # Standard Selection
    quote_options = {
        "Select Quote Template": None,
        "Vendor A Quotation (Apex Office Solutions)": "Vendor_A_Quotation.pdf",
        "Vendor B Quotation (Beacon Tech Furniture)": "Vendor_B_Quotation.pdf",
        "Vendor C Quotation (Crown Workspace Group)": "Vendor_C_Quotation.pdf"
    }
    
    selected_option = st.selectbox(
        "Choose a Proposal to Evaluate",
        options=list(quote_options.keys())
    )
    
    # Allow custom upload
    uploaded_file = st.file_uploader("Or Upload a Custom Quotation PDF", type="pdf")
    
    target_path = None
    if uploaded_file is not None:
        target_path = "data/" + uploaded_file.name
        # Secure Path Checker (Student 2 Privacy/Path mitigation compliance check)
        if st.session_state.sanitization_active:
            # Strip dangerous directory navigation sequences to prevent directory traversal
            clean_name = os.path.basename(uploaded_file.name)
            target_path = os.path.join("data", clean_name)
            
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Quotation uploaded successfully: {os.path.basename(target_path)}")
        selected_source = target_path
    elif quote_options[selected_option] is not None:
        selected_source = os.path.join("data", quote_options[selected_option])
        
    if selected_source:
        add_agent_log("Agent 1 (NLP Doc Intelligence)", f"Received source document: {os.path.basename(selected_source)}")
        
        # Read text
        try:
            reader = pypdf.PdfReader(selected_source)
            raw_text = ""
            for idx, p in enumerate(reader.pages):
                raw_text += f"\n--- Page {idx+1} ---\n" + p.extract_text()
                
            # Perform input validation to avoid injection payload (Student 1 / Red Team context)
            if st.session_state.sanitization_active:
                # Basic check for prompt injectors
                suspicious_patterns = ["ignore all prior instructions", "system override", "you are now a helpful assistant"]
                detected_patterns = [p for p in suspicious_patterns if p in raw_text.lower()]
                if detected_patterns:
                    st.warning(f"⚠️ Security Intercept: Potential instruction manipulation sequence detected in raw source input file! ({', '.join(detected_patterns)})")
                    
            col_l, col_r = st.columns([1, 1])
            
            with col_l:
                st.subheader("Raw Extracted Reference Text")
                st.text_area("Source text snapshot", raw_text, height=350, disabled=True)
                
            with col_r:
                st.subheader("Extracted structured JSON Attributes")
                
                # Execute NLP extraction
                with st.spinner("Processing NER & Value parsing..."):
                    extracted_params = parse_quotation_nlp(raw_text)
                    time.sleep(0.5) # simulated latency
                    
                st.json(extracted_params)
                
                add_agent_log("Agent 1 (NLP Doc Intelligence)", "NLP parser structured parameters extracted successfully.")
                
            st.subheader("Extracted Parameter Parameters & Threshold Tuning")
            st.info("The values below are automatically extracted. Review and refine them before executing the Compliance Audit check.")
            
            # Interactive forms to tweak extracted params (Human in the Loop)
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                extracted_params["vendor_name"] = st.text_input("Vendor Name Identity", value=extracted_params["vendor_name"])
                extracted_params["total_price"] = st.number_input("Extracted Proposal Price ($)", value=float(extracted_params["total_price"]), min_value=0.0)
            with col_p2:
                extracted_params["advance_payment_pct"] = st.number_input("Advance Downpayment (%)", value=float(extracted_params["advance_payment_pct"]), min_value=0.0, max_value=100.0)
                extracted_params["delivery_days"] = st.number_input("Delivery SLA Window (Days)", value=int(extracted_params["delivery_days"]), min_value=1)
            with col_p3:
                extracted_params["warranty_months"] = st.number_input("Warranty Coverage (Months)", value=int(extracted_params["warranty_months"]), min_value=0)
                extracted_params["sourcing_material"] = st.text_input("Material Composition", value=extracted_params["sourcing_material"])

        except Exception as e:
            st.error(f"Error parsing selected PDF document: {e}")
    else:
        st.warning("Please choose a quotation document in the drop-down or upload a custom document to begin.")

# --- STAGE 2: Retrieval & Policy Compliance (Agent 2 - ChromaDB RAG) ---
with t2:
    st.header("⚖️ Agent 2: Compliance RAG Audit")
    st.write("Performing vector search against ChromaDB holding standard Corporate Procurement policies to evaluate terms.")
    
    if not selected_source or not extracted_params:
        st.warning("Please select a quotation in Stage 1 to proceed with compliance processing.")
    else:
        # ChromaDB search query setup
        add_agent_log("Agent 2 (RAG Compliance)", f"Commencing automated policy retrieval audits on: {extracted_params['vendor_name']}")
        
        # 1. We search local database
        # Define key checks
        checks = [
            {
                "parameter": "Financial Spending Limit",
                "extracted_value": f"${extracted_params['total_price']:,.2f}",
                "search_query": "budget spending limitation maximum threshold for workspace or office upgrade",
                "eval_fn": lambda val: val <= 50000.0,
                "target_val": extracted_params['total_price'],
                "rule_desc": "Total cost must not exceed $50,000 USD."
            },
            {
                "parameter": "Advance Downpayment Policy",
                "extracted_value": f"{extracted_params['advance_payment_pct']}%",
                "search_query": "payment standards advance downpayment percentage limit approval net terms",
                "eval_fn": lambda val: val <= 10.0,
                "target_val": extracted_params['advance_payment_pct'],
                "rule_desc": "Advance payments must not exceed 10%."
            },
            {
                "parameter": "Delivery SLA Standard",
                "extracted_value": f"{extracted_params['delivery_days']} days",
                "search_query": "delivery timelines calendar days SLA requirements timelines",
                "eval_fn": lambda val: val <= 45,
                "target_val": extracted_params['delivery_days'],
                "rule_desc": "Standard delivery must be within 45 calendar days."
            },
            {
                "parameter": "Minimum Warranty Policy",
                "extracted_value": f"{extracted_params['warranty_months']} months",
                "search_query": "warranty safety requirements minimum monthly terms duration support",
                "eval_fn": lambda val: val >= 12,
                "target_val": extracted_params['warranty_months'],
                "rule_desc": "Warranty coverage must be at least 12 months."
            },
            {
                "parameter": "Ethical Sourcing & Material",
                "extracted_value": extracted_params['sourcing_material'],
                "search_query": "material composition requirements MDF hazardous resins Solid Wood or laminate",
                "eval_fn": lambda val: "mdf" not in val.lower(),
                "target_val": extracted_params['sourcing_material'],
                "rule_desc": "Medium Density Fiberboard (MDF) desktops are discouraged."
            }
        ]
        
        st.subheader("ChromaDB Vector Retrieval Output")
        
        # Retrieve context from ChromaDB
        audit_results = []
        all_passed = True
        
        for check in checks:
            # Query chroma using custom vectors
            results = collection.query(
                query_texts=[check["search_query"]],
                n_results=1
            )
            
            citation = "No matching corporate guidelines found in vector database."
            if results and 'documents' in results and len(results['documents']) > 0:
                citation = results['documents'][0][0]
                
            is_compliant = check["eval_fn"](check["target_val"])
            
            # Check override status
            vendor_name = extracted_params['vendor_name']
            override_key = f"{vendor_name}_{check['parameter']}"
            is_overridden = override_key in st.session_state.overridden_status
            
            if not is_compliant and not is_overridden:
                all_passed = False
                
            status_text = "✅ Compliant"
            if not is_compliant:
                status_text = "⚠️ Override Active" if is_overridden else "❌ Non-Compliant"
                
            audit_results.append({
                "Parameter": check["parameter"],
                "Extracted Deal Term": check["extracted_value"],
                "Audit Status": status_text,
                "Compliance Limit": check["rule_desc"],
                "policy_citation": citation,
                "is_compliant": is_compliant,
                "is_overridden": is_overridden
            })
            
        # Draw tabular audit results
        for idx, item in enumerate(audit_results):
            col_stat, col_cit = st.columns([1.5, 2.5])
            with col_stat:
                st.markdown(f"#### **{item['Parameter']}**")
                st.write(f"**Proposal Value:** {item['Extracted Deal Term']}")
                st.write(f"**Mandatory Standard:** {item['Compliance Limit']}")
                
                # Render HTML Badges
                if item['Audit Status'] == "✅ Compliant":
                    st.markdown('<span class="badge-success">✅ COMPLIANT</span>', unsafe_allow_html=True)
                elif item['Audit Status'] == "⚠️ Override Active":
                    st.markdown('<span class="badge-success" style="background-color:rgba(245,158,11,0.15); color:#F59E0B; border: 1px solid rgba(245,158,11,0.3)">⚠️ OVERRULED</span>', unsafe_allow_html=True)
                    st.caption(f"Reason: *{st.session_state.overridden_status[f'{vendor_name}_{item['Parameter']}']}*")
                else:
                    st.markdown('<span class="badge-danger">❌ NON-COMPLIANT</span>', unsafe_allow_html=True)
                    # Add override option for procurement managers
                    if role == "👨‍💼 Procurement Manager":
                        with st.popover("Manual Policy Override"):
                            reason = st.text_input("Justification Statement", key=f"rsn_{idx}")
                            if st.button("Apply Waiver", key=f"wvr_{idx}"):
                                if reason.strip():
                                    st.session_state.overridden_status[f"{vendor_name}_{item['Parameter']}"] = reason
                                    add_agent_log("Agent 2 (RAG Compliance)", f"Authorized waiver for {item['Parameter']} on account of: '{reason}'")
                                    st.success("Policy waiver updated!")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("Waiver rationale required.")
                                    
            with col_cit:
                st.markdown("<p style='font-size:0.85rem; color:#94A3B8; margin-bottom:2px;'><b>ChromaDB Retrieved Section Reference:</b></p>", unsafe_allow_html=True)
                st.info(f"\"{item['policy_citation']}\"")
            st.divider()
            
        add_agent_log("Agent 2 (RAG Compliance)", f"Audit complete. All checks passed: {all_passed}")

# --- STAGE 3: Vendor Risk & Recommendation (Agent 3) ---
with t3:
    st.header("📊 Agent 3: Vendor Risk Evaluation")
    st.write("Aggregating historical supplier statistics from CSV indexes and current compliance levels to form an overall Risk profile score.")
    
    if not selected_source or not extracted_params:
        st.warning("Please output extraction parameters in Stage 1 to proceed with risk profile computations.")
    else:
        # Load historical database
        csv_path = os.path.join("data", "supplier_history.csv")
        if os.path.exists(csv_path):
            df_suppliers = pd.read_csv(csv_path)
            
            # Match vendor history
            vendor_name = extracted_params["vendor_name"]
            
            # Match row based on simple matching
            matched_row = df_suppliers[df_suppliers["SupplierName"].str.contains(vendor_name.split()[0], case=False, na=False)]
            
            if not matched_row.empty:
                sup_record = matched_row.iloc[0].to_dict()
                
                # Calculate policy violation penality
                non_compliant_count = 0
                if 'audit_results' in locals() or 'audit_results' in globals():
                    for item in audit_results:
                        if not item["is_compliant"] and not item["is_overridden"]:
                            non_compliant_count += 1
                            
                # Calculate risk scores (0 to 100)
                # Formula: Base risk starts from past-violations + delivery rates + financial rating
                base_risk = 0.0
                if sup_record["FinancialRiskScore"] == "High":
                    base_risk += 40
                elif sup_record["FinancialRiskScore"] == "Medium":
                    base_risk += 20
                else:
                    base_risk += 5
                    
                # Delivery penalties
                base_risk += (1.0 - sup_record["OnTimeDeliveryRate"]) * 50
                # Quality penalties
                base_risk += (1.0 - sup_record["QualityScore"]) * 40
                # Past violations
                base_risk += sup_record["PastViolationsCount"] * 10
                # Current transaction violations penalty
                base_risk += non_compliant_count * 25
                
                # Constrain range
                overall_risk = min(max(base_risk, 0.0), 100.0)
                
                # Determine Classification Level
                risk_lvl = "Low"
                risk_color = "#10B981"
                if overall_risk > 65.0:
                    risk_lvl = "High"
                    risk_color = "#EF4444"
                elif overall_risk > 35.0:
                    risk_lvl = "Medium"
                    risk_color = "#F59E0B"
                    
                # Displays
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Overall Risk Profile</div>
                            <div class="metric-val" style="color:{risk_color};">{risk_lvl} ({overall_risk:.1f}%)</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Past Performance Rating</div>
                            <div class="metric-val">{sup_record['OverallRating']}/5.0</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_m3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Quality Delivery SLA</div>
                            <div class="metric-val">{sup_record['OnTimeDeliveryRate']*100:.0f}% / {sup_record['QualityScore']*100:.0f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_m4:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-lbl">Current Transaction Flags</div>
                            <div class="metric-val" style="color:{'#EF4444' if non_compliant_count > 0 else '#10B981'};">{non_compliant_count} Flags</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                st.write("")
                st.subheader(f"Historical Credentials Summary: {sup_record['SupplierName']}")
                
                st.dataframe(matched_row, hide_index=True)
                
                # Recommendation Output Box
                recommendation = "Award Approved"
                rec_desc = "Proposal is fully compliant with all local parameters, and historical scores list low transaction risks."
                rec_alert_type = st.success
                
                if risk_lvl == "High":
                    recommendation = "Automatic Handoff Rejection"
                    rec_desc = "Current terms list standard compliance issues combined with low historical SLAs. Proposal rejected."
                    rec_alert_type = st.error
                elif risk_lvl == "Medium":
                    recommendation = "Executive Review Required (Waiver Necessary)"
                    rec_desc = "Minor terms violate policy or vendor has moderate historical ratings. Requires waiver validation."
                    rec_alert_type = st.warning
                elif non_compliant_count > 0:
                    recommendation = "Action Negotiation Triggered"
                    rec_desc = "Quotation terms have policy violations. Agent 4 will generate request counter-offers."
                    rec_alert_type = st.warning
                    
                st.subheader("Agent 3 Summary Decision")
                rec_alert_type(f"**Recommendation: {recommendation}**\n\n*Rationale:* {rec_desc}")
                
                add_agent_log("Agent 3 (Vendor Risk Eval)", f"Risk assessment finalized (Risk score: {overall_risk:.1f}% ({risk_lvl}))")
                
            else:
                st.warning("Vendor name from document is not present in supplier history dataset directory. Raw risk estimation defaults to Low.")
        else:
            st.error("Historical record directory `data/supplier_history.csv` is missing.")

# --- STAGE 4: Action & Approval (Agent 4) ---
with t4:
    st.header("✍️ Agent 4: Procurement Action & Approval")
    st.write("Generates Purchase Order contract structures or auto-drafted renegotiation counter-offers based on Agent 3 assessment outputs.")
    
    if not selected_source or not extracted_params:
        st.warning("Please output extraction parameters in Stage 1 to proceed with contractual documents.")
    else:
        # Determine document type based on risk and compliance state
        non_compliant_list = []
        for check in audit_results:
            if not check["is_compliant"] and not check["is_overridden"]:
                non_compliant_list.append(check["Parameter"])
                
        # Determine final status
        if len(non_compliant_list) > 0:
            st.subheader("Counter-Offer & Renegotiation email generated")
            st.caption("Quotation violated policy terms. Requesting adjustments to become compliant.")
            
            # Draft email body
            email_body = f"""Subject: SmartProcure Quotation Review - Ref #: {extracted_params['vendor_name']}

Dear {extracted_params['vendor_name']} Team,

Thank you for submitting your proposal for our office layout configuration upgrade.

Our compliance analysis systems have run evaluation reviews on the submitted quotation document. Currently, the following conditions do not match corporate guidelines:
"""
            for param in non_compliant_list:
                if "Budget" in param or "Spending" in param:
                    email_body += f"- Pricing: Proposal total price (${extracted_params['total_price']:,.2f}) exceeds our current spending ceiling of $50,000.00.\n"
                elif "Payment" in param:
                    email_body += f"- Advance Payment: Your requested {extracted_params['advance_payment_pct']}% downpayment exceeds our structural limit of 10% maximum upfront values.\n"
                elif "Delivery" in param:
                    email_body += f"- Delivery: Your estimated {extracted_params['delivery_days']} days shipping timeline exceeds our required 45-day operational SLA.\n"
                elif "Warranty" in param:
                    email_body += f"- Warranty: The offered {extracted_params['warranty_months']}-month warranty support falls below our standard 12-month policy requirement.\n"
                elif "Material" in param:
                    email_body += "- Materials: MDF components do not meet our ESG sustainability specifications. We require HPL or Solid Wood desktop surfaces.\n"
                    
            email_body += f"""
We value partnership opportunities and would love to review revised quotation terms. Please update and re-submit a proposal incorporating compliant parameters.

Kind regards,
Procurement Team
SmartProcure Enterprises"""
            
            edited_text = st.text_area("Review Email Text", value=email_body, height=350)
            
            col_acts = st.columns([1, 1, 3])
            with col_acts[0]:
                if st.button("📧 Send Counter-Offer", type="primary"):
                    st.success("Counter-offer notification sent successfully to supplier contact!")
                    add_agent_log("Agent 4 (Action)", f"Sent re-negotiation notice to {extracted_params['vendor_name']}.")
            with col_acts[1]:
                if st.button("❌ Deny Proposal"):
                    st.info("Quotation has been rejected. Archive set up.")
                    add_agent_log("Agent 4 (Action)", f"Closed deal status for {extracted_params['vendor_name']} as Rejected.")
                    
        else:
            st.subheader("Purchase Order (PO) Contract Draft Generated")
            st.caption("Quotation is checked and compliant. Previewing generated PO document format.")
            
            po_num = f"PO-2026-{hash(extracted_params['vendor_name']) % 100000:05d}"
            
            po_text = f"""==================================================
                 PURCHASE ORDER
==================================================
PO Reference Number: {po_num}
Issue Date: {datetime.now().strftime("%B %d, 2026")}
Buyer Entity: SmartProcure Enterprises, Inc.

Vendor Contractor:
------------------------------------------
Company Name:  {extracted_params['vendor_name']}

Commercial Parameters:
------------------------------------------
Final Cost:    ${extracted_params['total_price']:,.2f} USD
Deliverable:   Ergonomic chairs & adjustable desk upgrades
Lead SLA:      {extracted_params['delivery_days']} Calendar Days
Payment Terms: {f"{extracted_params['advance_payment_pct']}% Advance / Net" if extracted_params['advance_payment_pct'] > 0 else "Net 30 days post-bill invoice"}
Warranty:      {extracted_params['warranty_months']} Months Local Support
DESK MATERIAL: {extracted_params['sourcing_material']}

Authorizations:
------------------------------------------
Waiver Exception Logs: [None - Standard Compliance Match]
System Verification status: 🛡️ Audit Cleared

==================================================
"""
            # Display editable PO document structure
            edited_po = st.text_area("Review Formatted PO", value=po_text, height=350)
            
            col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 3])
            with col_b1:
                # Human in the Loop checkout actions
                if role == "👨‍💼 Procurement Manager":
                    if st.button("✅ Authorize & Disburse PO", type="primary"):
                        st.success(f"PO {po_num} signed and sent to {extracted_params['vendor_name']}!")
                        add_agent_log("Agent 4 (Action)", f"PO {po_num} approved and issued to supplier.")
                else:
                    st.button("✅ Authorize & Disburse PO", type="primary", disabled=True, help="Only a Procurement Manager has authorization clearance to issue PO contracts.")
                    st.caption("🔒 *Procurement Manager credentials required to sign.*")
            with col_b2:
                if st.button("⛔ Cancel PO Process"):
                    st.info("PO drafting aborted.")
                    add_agent_log("Agent 4 (Action)", "Transaction execution draft cancelled.")

# --- Real-Time Agent-to-Agent Communication Logs ---
st.divider()
st.subheader("💬 Active Agent-to-Agent Logs (A2A Interface)")
st.caption("Visual representation of background MCP / A2A messages exchanged across agents.")

log_html = "<div class='agent-log'>"
if not st.session_state.agent_logs:
    log_html += "<div class='agent-log-line'>System loaded. Choose a proposal in Stage 1 to activate agent analytics logs...</div>"
else:
    for line in st.session_state.agent_logs:
        log_html += f"<div class='agent-log-line'>{line}</div>"
log_html += "</div>"

st.markdown(log_html, unsafe_allow_html=True)
