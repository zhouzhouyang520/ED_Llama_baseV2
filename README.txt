# Code and data for LLMs based on Empathetic-Dialogues dataset.

## Environment Installation

To run this code, you should：
1. Clone the project from github.
```sh
git clone https://github.com/zhouzhouyang520/ED_Llama_baseV2.git
```

2. Make the directory 'models' and put LLM model file into it:
```
mkdir models/
```

3. Enter the project root directory
```sh
cd ED_Llama_baseV2/examples
```

4. Download the data, and put all the files into the directory data/ed_json_data. [Baidu cloud link](https://pan.baidu.com/s/1iezsa7WMYDCR8bPYKMSsyg?pwd=9sr6) with Code: 9sr6 or [Google cloud link](https://drive.google.com/file/d/1SqlVfIn5lTlJZDnJAbgoxnY2OR7NAI5j/view?usp=drive_link)

5. Run the code.
   ```
   sh run.sh
   ```
6. Results will be saved in either ./out.log or ./output_ed/ed_result.txt

