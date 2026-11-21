import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


REPORT_FOLDER = "reports"


def generate_pdf_report(report_type, details):

    os.makedirs(REPORT_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"disaster_report_{timestamp}.pdf"
    filepath = os.path.join(REPORT_FOLDER, filename)

    document = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.darkblue,
        fontSize=19,
        spaceAfter=15
    )

    content = []

    content.append(
        Paragraph("MULTIMODAL DISASTER INTELLIGENCE PLATFORM", title_style)
    )

    content.append(Paragraph("Disaster Assessment Report", styles["Heading2"]))
    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            f"Report Type: {report_type}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 20))

    table_data = [["Parameter", "Result"]]

    for key, value in details.items():
        table_data.append([str(key), str(value)])

    table = Table(table_data, colWidths=[2.5 * inch, 3.5 * inch])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    content.append(table)
    content.append(Spacer(1, 25))

    content.append(
        Paragraph(
            "Note: This report contains AI-generated classification "
            "or historical risk assessment results. It is not an official "
            "emergency warning or a live disaster forecast.",
            styles["Italic"]
        )
    )

    document.build(content)

    return filename