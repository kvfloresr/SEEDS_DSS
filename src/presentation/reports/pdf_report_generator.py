from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Frame, PageTemplate
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from src.infrastructure.mongo_db import get_db
import gridfs
import io

def generate_pdf_in_memory(analysis_id):
    try:
        db = get_db()
        fs = gridfs.GridFS(db, collection="images_files")
        
        # Obtener datos
        analysis = db.analyses.find_one({"analysis_id": analysis_id})
        if not analysis:
            return None
        
        pdf_buffer = io.BytesIO()
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', fontSize=28, alignment=TA_CENTER, textColor=colors.darkgreen, spaceAfter=10, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle('Subtitle', fontSize=12, alignment=TA_CENTER, textColor=colors.green, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', fontSize=18, alignment=TA_LEFT, textColor=colors.darkgreen, spaceAfter=10, fontName='Helvetica-Bold')
        normal_style = ParagraphStyle('Normal', fontSize=12, alignment=TA_LEFT, spaceAfter=5, textColor=colors.black)
        footer_style = ParagraphStyle('Footer', fontSize=10, alignment=TA_CENTER, textColor=colors.grey)
        
        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 10)
            canvas.setFillColor(colors.grey)
            canvas.drawString(0.5*inch, 0.5*inch, f"Informe generado por Sistema DSS - Página {doc.page}")
            canvas.restoreState()
        
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, 
                        leftMargin=0.5*inch, rightMargin=0.5*inch, 
                        topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        
        story.append(Paragraph("Sistema de Análisis de Semillas DSS", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph("Informe Detallado de Verificación de Calidad", subtitle_style))
        story.append(Paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", normal_style))
        story.append(Spacer(1, 30))
        
        # Información General (Tabla Atractiva)
        story.append(Paragraph("Información del Análisis", heading_style))
        data = [
            ["ID de Análisis", analysis_id],
            ["Fecha de Procesamiento", analysis.get("processed_at", "N/A").strftime("%Y-%m-%d %H:%M") if analysis.get("processed_at") else "N/A"],
            ["Clase Predicha", analysis.get("predicted_class", "N/A")],
            ["Probabilidad", f"{analysis.get('probability', 0):.2f}%"]
        ]
        table = Table(data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.green),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Características Morfológicas
        story.append(Paragraph("🔍 Características Morfológicas Extraídas", heading_style))
        features = analysis.get("features", {})
        if isinstance(features, dict) and features:
            first_features = list(features.values())[0] if isinstance(list(features.values())[0], dict) else features
            morph_data = [["Parámetro", "Valor"]] + [[k, str(v)] for k, v in first_features.items()]
            morph_table = Table(morph_data, colWidths=[3*inch, 3*inch])
            morph_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.green),
            ]))
            story.append(morph_table)
        story.append(Spacer(1, 20))
        
        # Imágenes Overlay
        story.append(Paragraph("Imágenes Procesadas con Máscaras Aplicadas", heading_style))
        story.append(Paragraph("Verde: contorno de la semilla | Rojo: manchas y daños | Azul: impurezas", normal_style))
        for image_doc in db.images.find({"sample_guid": analysis["sample_guid"]}):
            if "overlay" in image_doc.get("filename", "").lower():
                file_data = fs.get(image_doc["gridfs_id"])
                img_buffer = io.BytesIO(file_data.read())
                img = Image(img_buffer, width=3.5*inch, height=2.5*inch)
                story.append(img)
                story.append(Paragraph(f"Imagen: {image_doc['filename']}", normal_style))
                story.append(Spacer(1, 15))
        
        # Notas
        story.append(Paragraph("Notas Adicionales", heading_style))
        review_notes = analysis.get("review_notes") or "Sin notas adicionales."
        story.append(Paragraph(review_notes, normal_style))
        

        
        # Generar PDF
        doc.build(story)
        return pdf_buffer
    except Exception as e:
        print(f"Error generando PDF en memoria: {e}")
        return None