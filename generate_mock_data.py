import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def create_pdf(filename, title, content_list):
    """Generates a professional styled PDF using ReportLab SimpleDocTemplate."""
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=54, leftMargin=54,
                            topMargin=54, bottomMargin=54)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0F766E'),
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#374151'),
        spaceAfter=8
    )
    
    label_style = ParagraphStyle(
        'DocLabel',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#111827'),
    )
    
    # Document Header
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 15))
    
    # Add items
    for item in content_list:
        item_type = item.get("type", "paragraph")
        
        if item_type == "paragraph":
            story.append(Paragraph(item["text"], body_style))
            story.append(Spacer(1, 4))
        
        elif item_type == "heading":
            story.append(Spacer(1, 10))
            story.append(Paragraph(item["text"], subtitle_style))
            story.append(Spacer(1, 5))
            
        elif item_type == "table":
            data = []
            # Check headers
            if "headers" in item:
                header_row = [Paragraph(str(h), label_style) for h in item["headers"]]
                data.append(header_row)
            
            for row in item["rows"]:
                row_data = []
                for val in row:
                    if isinstance(val, tuple) and len(val) == 2:
                        # (Label, text) key value pair style
                        lbl, txt = val
                        row_data.append(Paragraph(f"<b>{lbl}</b> {txt}", body_style))
                    else:
                        row_data.append(Paragraph(str(val), body_style))
                data.append(row_data)
                
            col_widths = item.get("widths", None)
            t = Table(data, colWidths=col_widths, hAlign='LEFT')
            
            t_style = TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ])
            
            if "headers" in item:
                # Add background for headers
                t_style.add('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3F4F6'))
                t_style.add('LINEBELOW', (0, 0), (-1, 0), 1.5, colors.HexColor('#D1D5DB'))
                
            t.setStyle(t_style)
            story.append(t)
            story.append(Spacer(1, 10))
            
        elif item_type == "spacer":
            story.append(Spacer(1, item.get("height", 10)))
            
    doc.build(story)
    print(f"Generated PDF: {filename}")

def build_data_files():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # 1. Procurement Policy PDF
    policy_content = [
        {"type": "paragraph", "text": "<b>Document Reference:</b> POL-PROC-2026-V1.0<br/><b>Effective Date:</b> January 1, 2026<br/><b>Approved By:</b> VP of Finance & Global Operations"},
        {"type": "spacer", "height": 10},
        {"type": "paragraph", "text": "This document outlines the mandatory compliance standards and criteria for all procurement activities related to office facilities, IT equipment, and enterprise renovations at SmartProcure Enterprises."},
        
        {"type": "heading", "text": "1. Financial Spending Thresholds"},
        {"type": "paragraph", "text": "The maximum allowable budget for department-wide workspace or office furniture upgrades is set strictly at <b>$50,000 USD</b>. Any quotation exceeding this maximum spending threshold will trigger an automatic compliance rejection and require direct authorization from the Chief Operating Officer (COO)."},
        
        {"type": "heading", "text": "2. Vendor Pricing and Payments"},
        {"type": "paragraph", "text": "Standard payment terms for SmartProcure Enterprises are net payment terms, specifically <b>Net 30</b> or <b>Net 60</b>. Advance payments are heavily discouraged to mitigate vendor risk. Upfront or advance payments must not exceed <b>10%</b> of the total Contract Value. Any advance payment request higher than 10% requires written approval from the Vice President (VP) of Finance."},
        
        {"type": "heading", "text": "3. Delivery SLA Standards"},
        {"type": "paragraph", "text": "Timely project execution is critical. Vendor delivery and installation schedules must be completed within a maximum of <b>45 calendar days</b> from the official Purchase Order (PO) issuance date. Proposals exceeding a 45-day delivery timeline will be flagged as high risk during evaluation."},
        
        {"type": "heading", "text": "4. Warranty & Support Clauses"},
        {"type": "paragraph", "text": "To guarantee asset durability and minimize maintenance overheads, all equipment, products, and services procured must come with a minimum warranty duration of <b>12 months</b> (1 Year). Extended warranty terms are highly preferred and will positively influence the vendor risk evaluation score."},
        
        {"type": "heading", "text": "5. Environmental and Sustainability Standards"},
        {"type": "paragraph", "text": "SmartProcure Enterprises is committed to carbon-neutral operations and environmentally friendly sourcing. Preference is explicitly given to suppliers holding verified environmental or quality management certifications: <b>ISO 14001</b> (Environmental Management Systems) or <b>BIFMA Certification</b> (Business and Institutional Furniture Manufacturers Association) or <b>ISO 9001</b> (Quality Management Systems)."},
        {"type": "paragraph", "text": "Furthermore, the utilization of medium-density fiberboard (MDF) contains hazardous resins and is discouraged for office work surfaces. Heavy-duty desktops must utilize sustainable Solid Wood or High-Pressure Laminate (HPL) materials to meet durability and environmental mandates."}
    ]
    create_pdf("data/Procurement_Policy_2026.pdf", "SmartProcure Corporate Procurement Policy (2026)", policy_content)

    # 2. Vendor A Quotation
    vendor_a_content = [
        {"type": "paragraph", "text": "<b>Quotation Reference:</b> APEX-2026-7890<br/><b>Date:</b> August 10, 2026<br/><b>Prepared By:</b> Apex Office Solutions Inc.<br/><b>Contact:</b> sales@apexoffice.com"},
        {"type": "spacer", "height": 10},
        {"type": "heading", "text": "Quotation Summary"},
        {"type": "paragraph", "text": "We are pleased to submit our formal proposal for the office setup upgrade. Our products represent high-quality materials manufactured in accordance with strict ISO quality standards."},
        
        {"type": "heading", "text": "Scope of Work & Pricing"},
        {"type": "table", "headers": ["Item Description", "Qty", "Unit Price ($)", "Total Price ($)"], "rows": [
            ["Ergonomic Task Chairs (Model: Apex-Comfort)", "100", "200.00", "20,000.00"],
            ["Motorized Standing Desks (Model: Apex-Rise, High-Pressure Laminate desktop)", "50", "500.00", "25,000.00"],
        ], "widths": [250, 50, 100, 100]},
        
        {"type": "table", "rows": [
            [("Total Proposal Price:", "$45,000.00 USD")]
        ], "widths": [500]},
        
        {"type": "heading", "text": "Commercial Terms and Conditions"},
        {"type": "table", "rows": [
            [("Payment Terms:", "Net 30 days after installation")],
            [("Delivery Timeline:", "30 calendar days from receipt of Purchase Order (PO)")],
            [("Warranty:", "12 Months (1 Year) local warranty covering all parts and motor controls")],
            [("Corporate Certifications:", "ISO 9001:2015 Quality Management Systems Certified")],
            [("Material Composition:", "Desktops are manufactured with high-stress High-Pressure Laminate (HPL) on birch plywood core.")]
        ], "widths": [500]}
    ]
    create_pdf("data/Vendor_A_Quotation.pdf", "Commercial Proposal - Apex Office Solutions", vendor_a_content)

    # 3. Vendor B Quotation (Violates several policies)
    vendor_b_content = [
        {"type": "paragraph", "text": "<b>Quotation Reference:</b> BTF-QT-9921<br/><b>Date:</b> August 12, 2026<br/><b>Prepared By:</b> Beacon Tech Furniture Ltd.<br/><b>Contact:</b> contact@beaconfurniture.co.uk"},
        {"type": "spacer", "height": 10},
        {"type": "heading", "text": "Formal Proposal for Workstation Upgrade"},
        {"type": "paragraph", "text": "Beacon Tech Furniture proposes a low-cost, budget-friendly workspace solution to meet the demands of layout modification."},
        
        {"type": "heading", "text": "Scope & Costs"},
        {"type": "table", "headers": ["Item Description", "Qty", "Unit Price ($)", "Total Price ($)"], "rows": [
            ["Standard Smart Chairs (Basic lumbar support)", "100", "150.00", "15,000.00"],
            ["Electric Sit-Stand Desks (Finished Medium Density Fiberboard (MDF) desktops)", "50", "460.00", "23,000.00"],
        ], "widths": [250, 50, 100, 100]},
        
        {"type": "table", "rows": [
            [("Net Bid Amount:", "$38,000.00 USD")]
        ], "widths": [500]},
        
        {"type": "heading", "text": "Contractual Terms"},
        {"type": "table", "rows": [
            [("Payment Schedule Terms:", "50% Advance Downpayment required; remaining 50% paid on final delivery and handoff.")],
            [("Estimated Lead-Time:", "60 calendar days (approximately 8 weeks) due to shipping and import delays.")],
            [("Warranty Coverage:", "6 Months standard parts and motor vendor warranty.")],
            [("Company Certifications:", "None listed or provided in standard format.")],
            [("Material Composition:", "Medium Density Fiberboard (MDF) desktops with standard wood veneer finish.")]
        ], "widths": [500]}
    ]
    create_pdf("data/Vendor_B_Quotation.pdf", "Proposal for Workspace Renovation - Beacon Tech Furniture", vendor_b_content)

    # 4. Vendor C Quotation (Premium supplier, higher budget)
    vendor_c_content = [
        {"type": "paragraph", "text": "<b>Quotation Reference:</b> CROWN-QS-2026-4412<br/><b>Date:</b> August 14, 2026<br/><b>Prepared By:</b> Crown Workspace Group<br/><b>Contact:</b> info@crownworkspace.com"},
        {"type": "spacer", "height": 10},
        {"type": "heading", "text": "High Performance Office Furnishing Bid"},
        {"type": "paragraph", "text": "Crown Workspace Group is pleased to offer a premium workspace furnishing solution. Our products are engineered for maximum ergonomic support and certified environmental sustainment, aligning with top ESG requirements."},
        
        {"type": "heading", "text": "Pricing Details"},
        {"type": "table", "headers": ["Standard Item Line", "Quantity", "Rate ($)", "Subtotal ($)"], "rows": [
            ["Crown Ergonomic Executive Chairs (Model: OrthoComfort, 3D Adjustments)", "100", "240.00", "24,000.00"],
            ["Crown Pro Dual-Motor Standing Desks (Sustainable Solid Oak desktop, 1.5 inch thick)", "50", "560.00", "28,000.00"],
        ], "widths": [250, 50, 100, 100]},
        
        {"type": "table", "rows": [
            [("Aggregate Proposal Price:", "$52,000.00 USD")]
        ], "widths": [500]},
        
        {"type": "heading", "text": "Proposal Conditions"},
        {"type": "table", "rows": [
            [("Payment Terms:", "Net 45 days post-billing invoice")],
            [("Delivery Timeline:", "15 calendar days from signed Purchase Order (PO)")],
            [("Warranty Guarantee:", "36 Months (3 Years) absolute comprehensive hardware and frame warranty support")],
            [("ESG Certifications:", "BIFMA Level 3 Sustainability Certification, ISO 14001:2015 Environmental, ISO 9001:2015 Quality Systems")],
            [("Material Sourcing:", "100% Sustainable Forest Stewardship Council (FSC) Solid Oak wood tables top")]
        ], "widths": [500]}
    ]
    create_pdf("data/Vendor_C_Quotation.pdf", "Corporate Furniture Supply Bid - Crown Workspace Group", vendor_c_content)

    # 5. supplier_history.csv
    supplier_history_data = {
        "SupplierID": ["VEND-A", "VEND-B", "VEND-C"],
        "SupplierName": ["Apex Office Solutions", "Beacon Tech Furniture", "Crown Workspace Group"],
        "OverallRating": [4.20, 2.80, 4.80],
        "OnTimeDeliveryRate": [0.92, 0.74, 0.98],
        "QualityScore": [0.90, 0.68, 0.96],
        "FinancialRiskScore": ["Low", "Medium", "Low"],
        "PastViolationsCount": [0, 2, 0],
        "YearFounded": [2010, 2021, 2005],
        "PreferredStatus": ["Yes", "No", "Yes"]
    }
    
    df = pd.DataFrame(supplier_history_data)
    df.to_csv("data/supplier_history.csv", index=False)
    print("Generated CSV: data/supplier_history.csv")

if __name__ == "__main__":
    build_data_files()
