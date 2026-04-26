import nltk

# Ensure NLTK data is available (with SSL bypass for macOS)
def _download_nltk_resources():
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for res in resources:
        nltk.download(res, quiet=True)

    print("NLTK resources downloaded successfully.")

_download_nltk_resources()