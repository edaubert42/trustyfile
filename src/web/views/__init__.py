"""HTML views of the web interface.

Every function here takes plain data and a language and returns an HTML
string. None of them knows which web framework displays the result, so the
same code serves the Streamlit page today and the server-rendered templates
tomorrow.
"""
