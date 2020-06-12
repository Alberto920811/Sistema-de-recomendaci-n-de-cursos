import io
import mysql.connector
import datetime
from layer_similarity import similarity
from preprocess import dirty_word, stop_words
from pprint import pprint
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

def extract_text(pdf_path):

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle)
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()
            yield text
            # close open handles
            converter.close()
            fake_file_handle.close()
            
def decode_text(pdf_path):

    pages = []
    page = ""
    n = 0
    pages = [page for page in extract_text(pdf_path)]
    for n in range(len(pages)):
        page += pages[n] + " "
    clean_text = [item.encode('utf-8') for item in page.split()]
    text_decoded = [it.decode('unicode-escape') for it in clean_text]
    return text_decoded

if __name__ == '__main__':

    #Realizar conexión a la base de datos 'mysql'
    mydb = mysql.connector.connect(host="localhost",
                                   user="root",passwd="",
                                   database="mysql",
                                   charset="utf8")
    cursor_CourseID = mydb.cursor()
    cursor_CourseID.execute("SELECT * FROM `courses`")
    CourseID = cursor_CourseID.fetchall()

    #Extraer columnas de base de datos 
    #y llamar al archivo de materia
    names = [row[1] for row in CourseID]
    dscrp = [row[2] for row in CourseID]
    obj   = [row[3] for row in CourseID]
    path  = 'examples/Mecanica de Materiales v2.pdf'
   
    #preprocesamiento de los datos
    names_c = dirty_word(names).clean_vector()
    obj_c   = dirty_word(obj).clean_vector()
    dscrp_c = dirty_word(dscrp).clean_vector()
    courses = [names_c[index]+" "+obj_c[index]+" "+dscrp_c[index] 
               for index in range(len(names))]
    file_class = dirty_word(decode_text(path)).clean_vector()
    #Realizar inferencia de similaridad
    percent_sim = similarity(names_c,file_class,courses).infer()
    pprint(percent_sim)
    exit()
    #Insertar datos a la tabla 'dataset'
    x = datetime.datetime.now()
    seperator = ' '
    words = seperator.join(file_class).encode('utf-8')
    dataset_update_cursor = mydb.cursor()
    sql = "INSERT INTO dataset (name, info, date) VALUES (%s, %s, %s)"
    val = (path, words, x)
    dataset_update_cursor.execute(sql, val)
    mydb.commit()
    print(dataset_update_cursor.rowcount, "dataset actualizado")