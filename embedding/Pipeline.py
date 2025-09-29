import os

class Pipeline:
    def __init__(self, input_folder_path :str, output_folder_path : str, config):
        self.config = config
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path

        # Ensure the output folder exists, create if not
        if not os.path.exists(self.output_folder_path):
            os.makedirs(self.output_folder_path)
