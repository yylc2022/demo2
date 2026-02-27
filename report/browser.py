import os

def show_notebook(html_path):
    from IPython.display import HTML
    HTML(filename=html_path)

def show_explorer(html_path):
    import webbrowser
    path = os.path.abspath(html_path)
    url = 'file://' + path
    webbrowser.open(url)
