import urllib.request

url_prefix = "http://sapling.stanford.edu/~zhihao/qasm_files/subst_history_"
url_suffix = ".qasm"

for i in range(40):
    url = url_prefix + str(i) + url_suffix
    save_filename = "subst_history_" + str(i) + ".qasm"
    try:
        urllib.request.urlretrieve(url, save_filename)
    except Exception as e:
        print(e)
