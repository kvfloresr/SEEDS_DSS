from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
from src.infrastructure.mongo_db import get_db
import gridfs
import io
import base64


def _img_from_b64(s, w, h):
    """RLImage desde un data-URL/base64 (los paneles por semilla)."""
    if not s:
        return None
    try:
        if "," in s:
            s = s.split(",", 1)[1]
        return RLImage(io.BytesIO(base64.b64decode(s)), width=w, height=h)
    except Exception:
        return None


def generate_pdf_in_memory(analysis_id):
    try:
        db = get_db()
        fs = gridfs.GridFS(db, collection="images_files")

        analysis = db.analyses.find_one({"analysis_id": analysis_id})
        if not analysis:
            return None

        iniaf = analysis.get("iniaf")

        pdf_buffer = io.BytesIO()

        title_style    = ParagraphStyle('TitleX', fontSize=20, alignment=TA_CENTER, textColor=colors.HexColor("#15803d"), spaceAfter=4, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle('SubX', fontSize=11, alignment=TA_CENTER, textColor=colors.HexColor("#16a34a"), spaceAfter=16)
        heading_style  = ParagraphStyle('HeadX', fontSize=14, alignment=TA_LEFT, textColor=colors.HexColor("#166534"), spaceBefore=10, spaceAfter=8, fontName='Helvetica-Bold')
        normal_style   = ParagraphStyle('NormX', fontSize=11, alignment=TA_LEFT, spaceAfter=5, textColor=colors.black)
        verdict_style  = ParagraphStyle('VerdX', fontSize=15, alignment=TA_CENTER, textColor=colors.white, fontName='Helvetica-Bold')
        cap_style      = ParagraphStyle('CapX', fontSize=8, textColor=colors.black, leading=10)
        src_style      = ParagraphStyle('SrcX', fontSize=8, textColor=colors.grey, spaceBefore=4)

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 9)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.5 * inch, 0.5 * inch, f"Informe SEEDS DSS - Página {doc.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                                topMargin=0.6 * inch, bottomMargin=0.7 * inch)
        story = []

        # ── Encabezado ──
        story.append(Paragraph("Sistema de Análisis de Semillas SEEDS DSS", title_style))
        story.append(Paragraph("", title_style))
        story.append(Paragraph("Informe de Verificación de Calidad — Norma INIAF", subtitle_style))
        story.append(Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", normal_style))
        story.append(Spacer(1, 14))

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
            ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 16))

        if iniaf:
            # ── Veredicto ──
            certifiable = bool(iniaf.get("certifiable"))
            level = iniaf.get("certification_level", "")
            reasons = iniaf.get("certification_reasons", []) or []
            bg = colors.HexColor("#16a34a") if certifiable else colors.HexColor("#dc2626")
            mark = "APTO" if certifiable else "NO APTO"
            vt = Table([[Paragraph(f"{mark} · {level}", verdict_style)]], colWidths=[6.8 * inch])
            vt.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), bg),
                                    ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12)]))
            story.append(vt)
            if not certifiable and reasons:
                story.append(Spacer(1, 4))
                story.append(Paragraph(f"<b>No cumple en:</b> {' · '.join(reasons)}", normal_style))
            story.append(Spacer(1, 14))

            # ── Indicadores ──
            indicators = iniaf.get("indicators", []) or []
            if indicators:
                story.append(Paragraph("Indicadores de Calidad Física (Tabla 3.1 INIAF)", heading_style))
                rows = [["Indicador", "Valor", "Norma INIAF", "Estado"]]
                fail_rows = []
                for i, ind in enumerate(indicators, start=1):
                    val = ind.get("value")
                    val_txt = f"{val}%" if isinstance(val, (int, float)) else str(val)
                    passed = bool(ind.get("pass"))
                    rows.append([ind.get("name", ""), val_txt, str(ind.get("threshold", "")), "Cumple" if passed else "No cumple"])
                    if not passed:
                        fail_rows.append(i)
                it = Table(rows, colWidths=[2.7 * inch, 1.2 * inch, 1.7 * inch, 1.2 * inch])
                style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
                    ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]
                for r in fail_rows:
                    style.append(('BACKGROUND', (0, r), (-1, r), colors.HexColor("#fee2e2")))
                    style.append(('TEXTCOLOR', (3, r), (3, r), colors.HexColor("#b91c1c")))
                it.setStyle(TableStyle(style))
                story.append(it)
                story.append(Spacer(1, 14))

            # ── Distribución ──
            dist = iniaf.get("distribution", []) or []
            if dist:
                story.append(Paragraph("Distribución por Categoría de Calidad (CNN)", heading_style))
                drows = [["Categoría", "Cantidad", "Porcentaje"]]
                for d in dist:
                    drows.append([d.get("label_es", d.get("class", "")), str(d.get("count", 0)), f"{d.get('percentage', 0)}%"])
                dt = Table(drows, colWidths=[3.6 * inch, 1.6 * inch, 1.6 * inch])
                dt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#166534")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#86efac")),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                    ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(dt)
                story.append(Spacer(1, 16))

            # ── Análisis morfológico por semilla (paneles guardados) ──
            per_seed = iniaf.get("per_seed", []) or []
            panels = [s for s in per_seed if s.get("panel")]
            if panels:
                story.append(Paragraph("Análisis Morfológico por Semilla", heading_style))
                story.append(Paragraph("Izquierda: recorte · Derecha: contorno detectado (verde).", normal_style))

                def seed_card(i, s):
                    img = _img_from_b64(s.get("panel"), 2.7 * inch, 1.33 * inch)
                    m = s.get("metrics") or {}
                    cap = Paragraph(
                        f"<b>#{i} {s.get('label_es', s.get('class',''))}</b> · {round(s.get('confidence', 0) * 100)}%<br/>"
                        f"Ø {m.get('diam_eq_px','-')} px · circ {m.get('circularidad','-')} · asp {m.get('aspecto','-')}",
                        cap_style)
                    inner = Table([[img if img else Paragraph("(sin imagen)", cap_style)], [cap]], colWidths=[2.8 * inch])
                    inner.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                            ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2)]))
                    return inner

                cards = [seed_card(i + 1, s) for i, s in enumerate(panels)]
                grid_rows = [cards[j:j + 2] for j in range(0, len(cards), 2)]
                if grid_rows and len(grid_rows[-1]) == 1:
                    grid_rows[-1].append("")
                grid = Table(grid_rows, colWidths=[3.0 * inch, 3.0 * inch])
                grid.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                        ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
                story.append(grid)
                story.append(Spacer(1, 14))
        else:
            story.append(Paragraph(
                "Este análisis no incluye datos de certificación INIAF (fue generado antes de habilitar el análisis por semilla).",
                normal_style))
            # Fallback: overlay viejo en GridFS
            for image_doc in db.images.find({"sample_guid": analysis.get("sample_guid")}):
                if "overlay" in image_doc.get("filename", "").lower():
                    file_data = fs.get(image_doc["gridfs_id"])
                    story.append(Spacer(1, 8))
                    story.append(RLImage(io.BytesIO(file_data.read()), width=4.0 * inch, height=2.8 * inch))
            story.append(Spacer(1, 12))

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