
# Truth-Conditional Captions for Time Series Data. EMNLP 2021


Code and Data for our EMNLP 2021 paper titled 'Truth-Conditional Captions for Time Series Data'. 


### Data
- processed_data/
- Contains
  - Synthetic time series data with NL annotations (pilot13final*) 
  - Stock time series data with NL annotations  (pilot16b*)

### Code
- allennlp_series/
- Experiment scripts: TODO
- Pre-trained model files: TODO


#### Requirements
- python=3.6
- pip install -r requirements.txt file
- Additional dependency on cocoevals
    - Download cocoevals code from following URL: https://drive.google.com/file/d/17AO2x_8_ltHI9WoVbcZIvGNZZ7wzKvFo/view?usp=sharing 
  - Move the folders to allennlp_series/training/metrics/



### Citation

```
@inproceedings{jhamtani2021truth,
  title={Truth-Conditional Captioning of Time Series Data},
  author={Jhamtani, Harsh and Berg-Kirkpatrick, Taylor},
  booktitle={EMNLP},
  year={2021}
}
```
