import kagglehub

# Download latest version
path = kagglehub.dataset_download("arunavakrchakraborty/covid19-twitter-dataset")

print("Path to dataset files:", path)