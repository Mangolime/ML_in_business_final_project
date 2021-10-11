# python-flask-docker
Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy
API: flask
Данные: с kaggle - https://archive.ics.uci.edu/ml/datasets/Adult

Задача:  Имеются данные о населении, полученные в результате переписи, в том числе о доходе. По таким признакам как возраст, образование, пол и т. д., известным о человеке, необходимо предсказывать уровень дохода - больше или меньше чем 50 тыс.

Используемые признаки:
 - age (int64)
 - workclass (object)
 - fnlwgt (int64)
 - education (object)
 - education-num (int64)
 - marital-status (object)
 - occupation (object)
 - relationship (object)
 - race (object)
 - sex (object)
 - capital-gain (int64)
 - capital-loss (int64)
 - hours-per-week (int64)
 - native-country (object)
 

Преобразования признаков: OneHotEncoder для категориальных признаков

Модель: RandomForestClassifier

### Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/fimochka-sudo/GB_docker_flask_example.git
$ cd GB_docker_flask_example
$ docker build -t fimochka/gb_docker_flask_example .
```

### Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -p 8181:8181 -v <your_local_path_to_pretrained_models>:/app/app/models fimochka/gb_docker_flask_example
```

### Переходим на localhost:8181
