# <img src="static/images/bouncerbench.svg" alt="BouncerBench icon" height="28"> BouncerBench
<div align="center">
  
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=for-the-badge)](https://wwww.bouncerbench.com)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge)](https://arxiv.org/abs/2506.17812)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/collections/uw-swag/bouncerbench-68570e7beb2f154502a92286)

</div>

# Making a Submission

This branch contains the logs, trajectories, and predictions of all leaderboard submissions. These instructions are adapted from [SWE-PolyBench](https://github.com/amazon-science/SWE-PolyBench). While we don't experiment with it, if your setup requires an execution environment please utilize the setup provided by [SWE-bench](https://www.swebench.com/SWE-bench/guides/evaluation/). To submit please follow the following procedure:

1. Fork the BouncerBench repository.

2. Clone the repository. Consider using `git clone --depth 1` if cloning takes too long.

3. Checkout the `submission` branch using:

```sh
git checkout submission
```

4. Create a file named `config.yaml` with details of your system based on the `example_config.yaml`. The config.yaml file should contain the following fields:
    - `name`: The name you want in the leaderboard entry
    - `oss`: `true` if your system is open-source
    - `site`: URL/link to more information about your system
    - `icon` : The name of the file in the `static/images` folder that contains the icon of your system. (Do not include the full path, just the file name)
    - `entries`: You can have  an entry to each of the three leaderboards, if you are only submitting to one leaderboard, you can remove the other two entries.
        - `lite`: The name of the file in the data folder that contains the output of your system for the lite leaderboard.
        - `input`: The name of the file in the data folder that contains the input of your system for the input bouncer leaderboard.
        - `output`: The name of the file in the data folder that contains the output of your system for the output bouncer leaderboard.

5. The files mentioned in the entries field should be placed in the data folder of your submission. (Do not include the full path, just the file name). The files should be structured as follows:
    - Submission to the input_bouncer leaderboard should be a json file with each task instance ID as the key and the value being a dictionary that must contain a boolean field `input_bounce` indicating whether the input was bounced or not. The dictionary can also contain other fields that your system outputs.
        - Example: `{"task_instance_id": {"input_bounce": true, "other_field": "value"}}`
    - Submission to the output_bouncer leaderboard should be a json file with each task instance ID as the key and the value being a dictionary that must contain a boolean field `output_bounce` indicating whether the output was bounced or not. The dictionary can also contain other fields that your system outputs.
        - Example: `{"task_instance_id": {"output_bounce": true, "other_field": "value"}}`
    - Submission to the lite leaderboard should be a json file with each task instance ID as the key and the value being a dictionary that must contains both `input_bounce` and `output_bounce` fields indicating whether the input and output were bounced or not. The dictionary can also contain other fields that your system outputs.
        - Example: `{"task_instance_id": {"input_bounce": true, "output_bounce": false, "other_field": "value"}}`

6. Additionally, you are required to submit reasoning traces for each task instance reflecting how your system solved the problem. The reasoning traces can be submitted in two ways:
    - If your traces can be represented as a simple json or a string, you can include them in the output json files under a field called `trace` for each task instance and the reasoning trace will be automatically extracted.
    - If your traces are in a different format, you should populate the traces folder with folders named after the task instance IDs, and each folder should contain files with the reasoning traces for that task instance. The folder structure should look like this:
        ```
        traces/
            task_instance_id_1/
                reasoning_trace_file_1.md
                reasoning_trace_file_2.json
            task_instance_id_2/
                reasoning_trace_file_1.yaml
        ```

7. Install the requirements and run the python script `evaluate.py` to validate your submission and evaluate it. This script will check if the config.yaml is valid, if the files mentioned in the config.yaml are present in the root folder, and if the traces folder is structured correctly. Finally, it will evaluate your submission against the BouncerBench dataset. You can run the script using:

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
python evaluate.py
```

8. If the output summary looks correct please create a pull request to the `submission` branch with the new folder created by the script. You must only have modifications in the `evaluation` (generated by the evaluate script) and `static` folders (for your icon). The other files are not meant to be committed.
```sh
git add .
git commit -m "your message"
git push origin submission
```
Please NOTE that you need to select `submission` as the `Base` branch and the `Compare` will be your forks `submission` branch.

## 📞 Have any doubts?
If you have any questions or doubts, please feel free to open an issue in the repository or reach out to us via email.