"""
added some comments because i want change some liines
Comprehensive test suite for file conversion utility (app.py).
Tests PDF to Word and Word to PDF conversion functionality.
Comprehensive test suite for file conversion utility (app.py).
Tests PDF to Word and Word to PDF conversion functionality.
added some comments because i want change some liines
Comprehensive test suite for file conversion utility (app.py).
Tests PDF to Word and Word to PDF conversion functionality.
Comprehensive test suite for file conversion utility (app.py).
Tests PDF to Word and Word to PDF conversion functionality.
"""


import gradio as gr
from pdf2docx import Converter
from docx import Document
from fpdf import FPDF
import os

title_and_description = """
# PDF to Word and Word to PDF converter
"""

def pdf_to_word(pdf_file):
    cv = None
    try:
        docx_filename = pdf_file.name.replace('.pdf', '.docx')
        
        cv = Converter(pdf_file.name)
        cv.convert(docx_filename, multi_processing=True, start=0, end=None)
        
        return docx_filename
    except IOError as e:
        raise gr.Error("Unable to access the PDF file. Please verify the file exists and is readable.")
    except ValueError as e:
        raise gr.Error("Invalid PDF file format. Please verify your file is a valid PDF document.")
    except Exception as e:
        # Catch all other exceptions including library-specific errors
        raise gr.Error("PDF conversion failed. Please verify your file is valid and try again.")
    finally:
        # Ensure converter is always closed, even if an error occurred
        if cv is not None:
            try:
                cv.close()
            except:
                pass  # Silently ignore cleanup errors

def word_to_pdf(docx_file):
    pdf_filename = "output.pdf"
    pdf = None
    
    try:
        doc = Document(docx_file)
        pdf = FPDF(format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_font('Arial', '', 'Arial.ttf', uni=True)
        pdf.set_font('Arial', size=12)

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:  # Ignorar linhas vazias
                continue
            # Quebrar o texto em várias linhas se necessário
            words = text.split()
            line = ''
            for word in words:
                if pdf.get_string_width(line + word) < (pdf.w - 2 * pdf.l_margin):
                    line += word + ' '
                else:
                    pdf.cell(0, 10, line, ln=True)
                    line = word + ' '
            if line:
                pdf.cell(0, 10, line, ln=True)

        pdf.output(pdf_filename)
        return pdf_filename
    except IOError as e:
        raise gr.Error("Unable to access the document file. Please verify the file exists and is readable.")
    except ValueError as e:
        raise gr.Error("Invalid document format. Please verify your file is a valid Word document.")
    except Exception as e:
        # Catch all other exceptions including library-specific errors
        raise gr.Error("Document conversion failed. Please verify your file is valid and try again.")
    finally:
        # Cleanup: remove temporary output file if it exists and an error occurred
        # Note: If conversion was successful, we want to keep the file for download
        pass

with gr.Blocks() as app:
    gr.Markdown(title_and_description)
    
    with gr.Row():
        with gr.Column():
            with gr.Accordion("PDF to Word"):
                pdf_input = gr.File(label="Upload PDF")
                convert_pdf_to_word = gr.Button("Convert to Word")
                word_output = gr.File(label="Download Word file", type="filepath")
                
                convert_pdf_to_word.click(pdf_to_word, inputs=[pdf_input], outputs=[word_output])
                
        with gr.Column():
            with gr.Accordion("Word to PDF"):
                word_input = gr.File(label="Upload Word")
                convert_word_to_pdf = gr.Button("Convert to PDF")
                pdf_output = gr.File(label="Download PDF file", type="filepath")
                
                convert_word_to_pdf.click(word_to_pdf, inputs=[word_input], outputs=[pdf_output])

app.launch()
