# creditcardfraud
for Datalab
# notebooks

1. EDA.ipynb: Um rápido análise dos dados fornecidos, dado que as classes não são balanceadas métricas como accuracy são substituidas por f1-score.
2. Train and hyperparameter tuning .ipynb: Com a finalidade de rodar varios algoritmos con diferentes parametros fue usado o GridSearch para salvar o melhor modelo o qual será usado posteriormente para fazer as predições.
3. Batch prediction with prefect.ipynb: Para a previsão em batch foi adotada a biblioteca prefect com a finalidade de executar a tarefa de predição periódicamente
4. Online prediction client.ipynb: Para a previsão online foi implementado uma API utilizando Flask como server que fica ativo esperando os requests
5. Online Prediction Server.ipynb: Para fazer o request foi usado Flask também.
6. requirements.txt: lista de requerimentos

