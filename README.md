# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institute is one of the educational institutions that has been established since 2000. Until now, it has produced many graduates with excellent reputations. However, there are also many students who do not complete their education, aka dropouts. 

This high number of dropouts is certainly a big problem for an educational institution. Therefore, Jaya Jaya Institute wants to detect as soon as possible students who might drop out so that they can be given special guidance.

### Business Problems
From business understanding section, we can define several questions that later will be answered in this project. The questions are:
1. What factors contribute most to a student's likelihood of dropping out?
2. What are the characteristics of students who are most at risk of dropping out compared to those who successfully graduate?

### Project Scope
The things we will do in this project include: 
1. Developing a dashboard to answer several questions defined on business problems. 
2. Developing a machine learning system to predict whether students with certain criteria have the potential to graduate or drop out.

### Preparation

Dataset: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

#### Setup environment (machine-learning-streamlit-app):
1. Create virtual environment

```
python -m venv streamlit-app
```

2. Activate the virtual environment

- Windows

```
streamlit-app\Scripts\activate.bat
```

or

```
streamlit-app\Scripts\Activate.ps1
```

or

```
streamlit-app\Scripts\activate
```

- Linux\Mac

```
source streamlit-app\Scripts\activate
```

3. Install library and dependency

```
pip install requirements_app.txt
```

## Run Machine Learning App

```
streamlit run app.py
```

Streamlit Community Cloud Link: 

## Business Dashboard
Based on Dashboard and EDA result, here are some interesting informations we can take,
1. Previous qualification grade and admission grade value for students who drop out tend to be slightly 2.lower. (notebook)
2. The age of students who dropout when they enrolled also tend to be much older (dashboard)
3. In the first semester, students who drop out also tend to have fewer classes approved as well as lower overall semester grades. (notebook)
4. Based on the proportion, students who attend class during evening has more likely to dropout. However, most students attend class in daytime and it also has bigger number of dropout comparing to evening. (notebook)
5. Students who are not displaced students are most likely to dropout based on proportion. They also have bigger number on dropout compared to displaced student. (notebook and dashboard)
6. Student that has educational special needs has slightly greater chance to dropout. However, this column has imbalance data and in numbers, student who doesnt have special needs is much bigger than who does. (notebook)
7. Student who is debtor also most likely to dropout and less likely to graduate based on proportion. It also has correlation with tuition fee up to date where students whose tuition fees is not up to date are most likely to dropout. (notebook and dashboard)
8. Male students also most likely to dropout based on proportion. However, both female and male have around same number of dropout students. (notebook and dashboard)
9. Based on proportion, students who are not scholarship holder and not international students are most likely to dropout. (notebook and dashboard)
10. Students who have marital status as married or divorced are most likely to dropout. (notebook and dashboard)
11. For application mode, students who applied over 23 years old are most likely to dropout. (notebook)

Dashboard Link: https://lookerstudio.google.com/reporting/3a013f84-ea2d-4388-bc98-69490c51bd9e

## Conclusion
The potential for students to drop out at the Jaya Jaya Institute can be influenced by several factors, starting from academic factors before and after enrolling; demographic factors such as older age, non-scholarship, and non-international students; social and economic factors such as debtor, non-displaced, etc. A machine learning system has also been created and deployed on the streamlit cloud using the Gradient Boosting algorithm to detect a student's risk of dropping out based on the criteria mentioned with the accuracy of 68%.


### Rekomendasi Action Items
Based on the result, here are the recommended actions for Jaya Jaya Institute:
1. Identify students with low admission and qualification grade from the start and monitor them in their first year.
2. Create special program for older and students with certain marital status, such as flexible schedule, sharing and discussion session, etc.
3. Identify students who are struggling financially and socially, and hold consultation sessions for them.
4. Support students with educational special needs, create access to certain facilities, education counselors, and psychological support.
5. Evaluate class both daytime and evening, starting from learning material, facility, or schedule.
6. Create special program for students, both female and male students, it could be like increasing their involvement to practical studies, or support and motivation for their own group.