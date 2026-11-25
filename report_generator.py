import os
import re
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Engineering Drawing Validation Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    """Removes emojis and special characters to prevent FPDF encoding errors."""
    if not text: return ""
    text = re.sub(r'\*\*', '', text) 
    text = text.replace('⚠️', '-').replace('✅', '-').replace('❌', '-')
    text = text.replace('₹', 'Rs. ').replace('²', '2').replace('³', '3')
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(est_area, det_area, overall_dev, status, est_cost, recs, fig_img_path):
    pdf = PDFReport()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. Validation Summary", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Validation Status: {clean_text(status)}", 0, 1)
    pdf.cell(0, 8, f"Detected Area: {det_area:.2f} sq m", 0, 1)
    pdf.cell(0, 8, f"Estimated Area: {est_area:.2f} sq m", 0, 1)
    pdf.cell(0, 8, f"Overall Deviation: {overall_dev:.2f}%", 0, 1)
    pdf.cell(0, 8, f"Estimated Cost: Rs. {est_cost:,.2f}", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Recommendations", 0, 1)
    pdf.set_font("Arial", size=11)
    if status == "Valid":
        pdf.multi_cell(0, 8, "Your estimation is within acceptable tolerances.")
    else:
        for rec in recs:
            if rec:
                pdf.multi_cell(0, 8, clean_text(rec))
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Visual Comparison", 0, 1)
    if os.path.exists(fig_img_path):
        pdf.image(fig_img_path, x=30, w=150)
        
    return pdf.output(dest='S').encode('latin-1')