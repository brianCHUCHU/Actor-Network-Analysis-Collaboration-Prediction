# Graph Neural Network for Movie Co-star Recommendation

This is a student Term Project in `Social Media Analysis` by Prof. Chih-Ping Wei, National Taiwan University.

## Quick Links

- [Social Media Analysis Term Project](https://github.com/brianCHUCHU/Social_Advertisement_Recommendation_System/blob/main/IRTM%20Term%20Project.pdf): For detailed documentation in formal PDF format.
- [Slides](): Presentation slides.

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
| Contributor | Work |
|-------------|------|
| *Jih-Ming Bai* | Problem Formulation, Model, Experiment and Analysis |
| *Cheng-Yu Kuan* | Literature Review, Gradio Demo   |
| *Po-Yen Chu*    | EGES Model and Experiment       |
| *Shang-Qing Su* | Data Collection, Report Delivery |
| *Chia-Shan Li*  | Data Collection, Report Delivery |

