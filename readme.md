# Information Retrive

# [Dataset](https://drive.google.com/drive/folders/1sKPgdemCLT35eVPy4IoHLvBjQJtzTXY9?usp=sharing)

## 1. Simple BM25 (with only TF-IDF)
| Metrics      | Context | question|
| -----------  | --------|---------|
| R@1          | 0.82    | 0.2     |
| R@5          | 0.91    | 0.395   |
| R@10         | 0.93    | 0.515   |
| R@20         | 0.935   | 0.6     |
| R@50         | 0.955   | 0.71    |
| R@100        | 0.98    | 0.78    |
| R@200        | 0.985   | 0.845   |
| R@500        | 0.985   | 0.87    |
| R@1000       | 0.99    | 0.895   |

## 2. BM25 and Pairwise Ranking
| Metrics      |  BM25 | BM25+rank |
| -----------  | ----- |-------- | 
| R@1          |  0.2  | 0.315   | 
| R@5          |  0.395| 0.575   |        
| R@10         |  0.515| 0.655   |         
| R@20         |  0.6  | 0.7     |   
| R@50         |  0.71 | 0.765   |   
| R@100        |  0.78 | 0.81    |  
| R@200        |  0.845| 0.84    |   
| R@500        |  0.87 | 0.87    |  
| R@1000       |  0.895| 0.885   |   

