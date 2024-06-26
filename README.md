# Graph Neural Network for Movie Co-star Recommendation

This is a student Term Project in `Social Media Analysis` by Prof. Chih-Ping Wei, National Taiwan University.

## Quick Links

- [Social Media Analysis Term Project](https://github.com/brianCHUCHU/Actor-Network-Analysis-Collaboration-Prediction/blob/main/Social_Media_Analysis_Term_Project.pdf): For detailed documentation in formal PDF format.
- [Slides](https://github.com/brianCHUCHU/Actor-Network-Analysis-Collaboration-Prediction/blob/main/Social_Media_Analysis_Term_Project_Slides.pdf): Presentation slides.

## Brief Description

### Abstract
Our research focuses on a data-driven approach to recommend movie co-stardom in perspective of profitability. We collected movie and actor data to construct collaboration network, and utilized node embedding (*EGES*, *Node2Vec*) and message passing machanism (*GCN*, *SEAL*) to solve a link prediction task.

### Model & Experiments

We conducted experiments on the collaboration network data of movie actors using
four models, which include:

***1) Baseline model*** (ML-based): only using features of the
two actors and predicting with XGBoost Classifier, 

***2) Benchmark models*** (EGES,
GCN): utilizing actor and network information, 

***3) Best model*** (SEAL): employing a more advanced model architecture for actor collaboration link prediction. 

We used AUC as the model evaluation metric.

|       | ML-based |  EGES  | GCN | SEAL |
|-------|----------|--------|-----|------|
| Valid |   61.1%  |  59.2% | 67.2% | 84.5% |
| Test  |   55.3%  |  59.2% | 67.6% | 80.1% |

Contribution
-----------
| **ID**       | **Member**       | **Work**                                       |
|--------------|------------------|------------------------------------------------|
| R11725041    | Jih-Ming Bai     | Literature Research, Models and Experiment     |
| R12725028    | Cheng-Yu Kuan    | Literature Research, Demo                      |
| B10704031    | Po-Yen Chu       | Literature Research, EGES (Model Building, Experiment) |
| B10704096    | Shang-Qing Su    | Literature Research, Data Collection           |
| B10704051    | Chia-Shan Li     | Literature Research, Data Collection, Slides   |
