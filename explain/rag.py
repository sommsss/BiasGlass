import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def rag_explain(metrics, scenario="", mitigated_metrics=None):
    dp = metrics["demographic_parity_difference"]
    eo = metrics["equalized_odds_difference"]
    di = metrics["disparate_impact_ratio"]
    groups = metrics["group_selection_rates"]
    bias_score = metrics["bias_risk_score"]
    bias_label = metrics["bias_risk_label"]
    accuracy = metrics["overall_accuracy"]

    mitigation_block = ""
    if mitigated_metrics:
        dp_m = mitigated_metrics["demographic_parity_difference"]
        eo_m = mitigated_metrics["equalized_odds_difference"]
        score_m = mitigated_metrics["bias_risk_score"]
        mitigation_block = f"""
After Bias Mitigation (ExponentiatedGradient):
- Demographic Parity Difference: {dp_m}
- Equalized Odds Difference: {eo_m}
- New Bias Risk Score: {score_m}/100
"""

    prompt = f"""
You are a senior AI fairness auditor producing a professional audit report.

SCENARIO: {scenario}
OVERALL MODEL ACCURACY: {accuracy}
BIAS RISK SCORE: {bias_score}/100 ({bias_label})

FAIRNESS METRICS (Before Mitigation):
- Demographic Parity Difference: {dp}
  (Acceptable range: |DP| < 0.1. Values above 0.2 are high concern.)
- Equalized Odds Difference: {eo}
  (Acceptable range: |EO| < 0.1)
- Disparate Impact Ratio: {di}
  (Legal standard: above 0.8. Below 0.8 = potential legal liability under EEOC 80% rule.)

GROUP SELECTION RATES:
{groups}

{mitigation_block}

REFERENCE CONTEXT:
- The EU AI Act (2024) classifies hiring, credit, and criminal justice systems as HIGH RISK — bias audits are legally required.
- The EEOC 80% rule: if the selection rate for a protected group is less than 80% of the highest group's rate, it signals adverse impact.
- Fairness is context-dependent: Demographic Parity and Equalized Odds can conflict. A model satisfying one may violate the other.
- Not all bias originates from the model — some reflects historical data inequality.

IMPORTANT INSTRUCTIONS:
- Use the actual numbers. Never be vague.
- Distinguish between model-introduced bias and data-driven disparity.
- Mention legal implications if the Disparate Impact Ratio is below 0.8.
- Be direct about severity.

Format your response EXACTLY like this:

### 🔍 Executive Summary
(2–3 lines: overall bias severity, most affected group, and whether the model is safe to deploy)

### 📊 Metric Interpretation
- Demographic Parity ({dp}): [explain what this means for real people]
- Equalized Odds ({eo}): [explain what this means for real people]
- Disparate Impact Ratio ({di}): [explain legal implications]

### 👥 Group Comparison
[Compare groups explicitly with numbers. Who is disadvantaged and by how much?]

### ⚖️ Model Bias vs Data Bias
[Is the disparity coming from the model, or from historical patterns in the data? Explain clearly.]

### 🚨 Risk Assessment
[Is this model safe to deploy? What are the real-world harms if it is?]

### 🔧 Recommended Fixes
1. [Specific fix with technique name]
2. [Specific fix with technique name]
3. [Specific fix with technique name]
4. [Specific fix with technique name]

### 📋 Compliance Note
[Comment on EEOC 80% rule and EU AI Act compliance based on the metrics]
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


def generate_pdf_report(metrics, scenario, explanation, mitigated_metrics=None):
    """
    Generates a PDF audit report using reportlab.
    Returns bytes of the PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        import io, datetime

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=22, textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6
        )
        story.append(Paragraph("BiasGlass — AI Fairness Audit Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | Scenario: {scenario}",
            styles["Normal"]
        ))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#4f46e5")))
        story.append(Spacer(1, 0.4*cm))

        # Bias Risk Score
        score = metrics["bias_risk_score"]
        label = metrics["bias_risk_label"]
        color_map = {"Low": "#16a34a", "Moderate": "#d97706", "High": "#dc2626"}
        score_color = colors.HexColor(color_map.get(label, "#000000"))

        score_style = ParagraphStyle(
            "Score", parent=styles["Normal"],
            fontSize=16, textColor=score_color, spaceAfter=4
        )
        story.append(Paragraph(f"Bias Risk Score: {score}/100 — {label}", score_style))
        story.append(Spacer(1, 0.3*cm))

        # Metrics Table
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                   fontSize=13, textColor=colors.HexColor("#1a1a2e"))
        story.append(Paragraph("Fairness Metrics", h2_style))

        rows = [["Metric", "Value", "Acceptable Threshold"]]
        rows.append(["Demographic Parity Difference", str(metrics["demographic_parity_difference"]), "< 0.1"])
        rows.append(["Equalized Odds Difference", str(metrics["equalized_odds_difference"]), "< 0.1"])
        rows.append(["Disparate Impact Ratio", str(metrics["disparate_impact_ratio"]), "> 0.8 (legal)"])
        rows.append(["Overall Accuracy", str(metrics["overall_accuracy"]), "—"])

        if mitigated_metrics:
            rows.append(["DP Difference (After Mitigation)", str(mitigated_metrics["demographic_parity_difference"]), "< 0.1"])
            rows.append(["EO Difference (After Mitigation)", str(mitigated_metrics["equalized_odds_difference"]), "< 0.1"])
            rows.append(["Bias Risk Score (After Mitigation)", str(mitigated_metrics["bias_risk_score"]) + "/100", "—"])

        table = Table(rows, colWidths=[7*cm, 4*cm, 5*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.5*cm))

        # LLM Explanation — strip markdown symbols for PDF
        story.append(Paragraph("AI-Generated Analysis", h2_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
        story.append(Spacer(1, 0.2*cm))

        body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                    fontSize=10, leading=15, spaceAfter=6)

        for line in explanation.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.15*cm))
                continue
            if line.startswith("### "):
                story.append(Paragraph(line.replace("### ", ""), h2_style))
            elif line.startswith("- "):
                story.append(Paragraph("• " + line[2:], body_style))
            else:
                story.append(Paragraph(line, body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    except ImportError:
        return None