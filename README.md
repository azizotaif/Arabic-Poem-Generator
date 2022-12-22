# Arabic Poem Generator

End-to-End Arabic poem generation project using GPT2

## Dataset
The model is trained on 55K Arabic poems from various categories, countries and eras

[Arabic Poetry Dataset](https://www.kaggle.com/datasets/ahmedabelal/arabic-poetry)

## Trained Model
[Download Trained GPT2 Model (1.4 GB)](https://drive.google.com/file/d/1shqlW9HDhIokOzHIrbNZ3aaVT21kXtyL/view?usp=sharing)

## Deploy

- Clone this repository and install the requirements
``` shell
git clone https://github.com/azizotaif/Arabic-Poem-Generator.git
pip install -r Arabic-Poem-Generator/requirements.txt
```

- Download the [trained model](https://drive.google.com/file/d/1shqlW9HDhIokOzHIrbNZ3aaVT21kXtyL/view?usp=sharing) and place it inside the model directory
``` shell
Arabic-Poem-Generator/model/
```

- Run the Flask web app and pass the following arguments

        1. --port : A port number, the default port is 5000

        2. --device : Inference device (CPU or CUDA), the default device is CPU
``` shell
python app.py --port 7000 --device "cuda"
```

- Open a web browser and enter your IP and the port specified in the previous step
Example:
``` shell
192.168.1.100:7000
```

- The following page will appear, write a line of Arabic poem and click Generate Poem

![image](https://drive.google.com/uc?export=view&id=1jfU_WDtxBX3Gyl97ubm-bgeqdT6uxG8n)

![image](https://drive.google.com/uc?export=view&id=1zD8X7je60OujuUYhCz7ruLPhMzrO5OjB)
