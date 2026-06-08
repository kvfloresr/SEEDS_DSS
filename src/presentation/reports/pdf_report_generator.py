from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from src.infrastructure.mongo_db import get_db
import gridfs
import io


def generate_pdf_in_memory(analysis_id):
    try:
        db = get_db()
        fs = gridfs.GridFS(db, collection="images_files")

        analysis = db.analyses.find_one({"analysis_id": analysis_id})
        if not analysis:
            return None

        iniaf = analysis.get("iniaf")  # resumen guardado desde el frontend (puede faltar en análisis viejos)

        pdf_buffer = io.BytesIO()

        title_style    = ParagraphStyle('TitleX', fontSize=26, alignment=TA_CENTER, textColor=colors.HexColor("#15803d"), spaceAfter=8, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle('SubX', fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor("#16a34a"), spaceAfter=18)
        heading_style  = ParagraphStyle('HeadX', fontSize=15, alignment=TA_LEFT, textColor=colors.HexColor("#166534"), spaceBefore=10, spaceAfter=8, fontName='Helvetica-Bold')
        normal_style   = ParagraphStyle('NormX', fontSize=11, alignment=TA_LEFT, spaceAfter=5, textColor=colors.black)
        verdict_style  = ParagraphStyle('VerdX', fontSize=16, alignment=TA_CENTER, textColor=colors.white, fontName='Helvetica-Bold')

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.5 * inch, 0.5 * inch, f"Informe generado por Sistema SEEDS DSS - Página {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                                topMargin=0.6 * inch, bottomMargin=0.7 * inch)
        story = []

        # ── Encabezado ──
        story.append(Paragraph("Sistema de Análisis de Semillas SEEDS DSS", title_style))
        story.append(Paragraph("Informe de Verificación de Calidad — Norma INIAF 2022", subtitle_style))
        story.append(Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", normal_style))
        story.append(Spacer(1, 16))

        # ── Información del análisis ──
        story.append(Paragraph("Información del Análisis", heading_style))
        proc = analysis.get("processed_at")
        info = [
            ["ID de Análisis", str(analysis_id)],
            ["Fecha de Procesamiento", proc.strftime("%Y-%m-%d %H:%M") if hasattr(proc, "strftime") else "N/A"],
            ["Semillas analizadas", str(iniaf.get("total_analyzed", "N/A")) if iniaf else "N/A"],
        ]
        t = Table(info, colWidths=[2.4 * inch, 4.4 * inch])
        t.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#dcfce7")),
            ('BACKGROUND', (1, 0), (1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 18))

        if iniaf:
            # ── Veredicto de certificación ──
            certifiable = bool(iniaf.get("certifiable"))
            level = iniaf.get("certification_level", "")
            reasons = iniaf.get("certification_reasons", []) or []
            bg = colors.HexColor("#16a34a") if certifiable else colors.HexColor("#dc2626")
            mark = "APTO" if certifiable else "NO APTO"
            verdict_tbl = Table([[Paragraph(f"{mark} · {level}", verdict_style)]], colWidths=[6.8 * inch])
            verdict_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ]))
            story.append(verdict_tbl)
            if not certifiable and reasons:
                story.append(Spacer(1, 4))
                story.append(Paragraph(f"<b>No cumple en:</b> {' · '.join(reasons)}", normal_style))
            story.append(Spacer(1, 16))

            # ── Tabla de indicadores INIAF ──
            indicators = iniaf.get("indicators", []) or []
            if indicators:
                story.append(Paragraph("Indicadores de Calidad Física (Tabla 3.1 INIAF)", heading_style))
                rows = [["Indicador", "Valor", "Norma INIAF", "Estado"]]
                fail_rows = []
                for i, ind in enumerate(indicators, start=1):
                    val = ind.get("value")
                    val_txt = f"{val}%" if isinstance(val, (int, float)) else str(val)
                    passed = bool(ind.get("pass"))
                    rows.append([ind.get("name", ""), val_txt, str(ind.get("threshold", "")),
                                 "Cumple" if passed else "No cumple"])
                    if not passed:
                        fail_rows.append(i)
                it = Table(rows, colWidths=[2.7 * inch, 1.2 * inch, 1.7 * inch, 1.2 * inch])
                style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]
                for r in fail_rows:
                    style.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor("#fee2e2")))
                    style.append(('TEXTCOLOR', (3, r), (3, r), colors.HexColor("#b91c1c")))
                it.setStyle(TableStyle(style))
                story.append(it)
                story.append(Paragraph(
                    "Fuente: Compendio de Normas Nacionales sobre Semillas de Especies Agrícolas, INIAF 2022.",
                    ParagraphStyle('src', fontSize=8, textColor=colors.grey, spaceBefore=4)))
                story.append(Spacer(1, 16))

            # ── Distribución por categoría ──
            dist = iniaf.get("distribution", []) or []
            if dist:
                story.append(Paragraph("Distribución por Categoría de Calidad (CNN)", heading_style))
                drows = [["Categoría", "Cantidad", "Porcentaje"]]
                for d in dist:
                    drows.append([d.get("label_es", d.get("class", "")),
                                  str(d.get("count", 0)), f"{d.get('percentage', 0)}%"])
                dt = Table(drows, colWidths=[3.6 * inch, 1.6 * inch, 1.6 * inch])
                dt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(dt)
                story.append(Spacer(1, 18))
        else:
            story.append(Paragraph(
                "Este análisis no incluye datos de certificación INIAF "
                "(fue generado antes de habilitar el análisis por semilla).",
                normal_style))
            story.append(Spacer(1, 12))

        # ── Imágenes de detección (overlay en GridFS) ──
        story.append(Paragraph("Detección Individual de Semillas", heading_style))
        story.append(Paragraph("Verde: semilla de soya detectada · Rojo: objeto no soya / impureza", normal_style))
        found_img = False
        for image_doc in db.images.find({"sample_guid": analysis.get("sample_guid")}):
            if "overlay" in image_doc.get("filename", "").lower():
                file_data = fs.get(image_doc["gridfs_id"])
                img_buffer = io.BytesIO(file_data.read())
                story.append(Spacer(1, 6))
                story.append(Image(img_buffer, width=4.2 * inch, height=3.0 * inch))
                found_img = True
        if not found_img:
            story.append(Paragraph("No hay imágenes de detección disponibles.", normal_style))
        story.append(Spacer(1, 16))

        # ── Notas ──
        story.append(Paragraph("Notas Adicionales", heading_style))
        story.append(Paragraph(analysis.get("review_notes") or "Sin notas adicionales.", normal_style))

        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        return pdf_buffer
    except Exception as e:
        print(f"Error generando PDF en memoria: {e}")
        import traceback
        traceback.print_exc()
        return None