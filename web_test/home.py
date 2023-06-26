import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("./data/NetflixOriginals.csv", encoding='latin1')






df['Genres'] = df['Genre'].str.split('/').apply(lambda genres: [genre.strip().lower() for genre in genres])

df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('science fiction', 'sci_fi') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('musicial', 'musical') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('making-of', 'making_of') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('romantic', 'romance') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('biopic', 'biographical') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('black comedy', 'dark_comedy') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('dark comedy', 'dark_comedy') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('mentalism special', 'supernatural') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('one-man show', 'one_man_show') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('stop motion', 'stop_motion') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('teen ', 'teenage ') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('hidden-camera prank ', '') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('coming-of-age', 'coming_of_age') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('variety show', 'variety_show') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('animated', 'animation') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace('anime', 'animation') for genre in genres])
df['Genres'] = df['Genres'].apply(lambda genres: [genre.replace(' film', '') for genre in genres])

df['Genres'] = df['Genres'].apply(lambda genres: [word for genre in genres for word in genre.split('-')])
df['Genres'] = df['Genres'].apply(lambda genres: [word for genre in genres for word in genre.split(' ')])

unique_genres = sorted(df['Genres'].explode().unique())
num_unique_genres = len(unique_genres)

print(num_unique_genres)
print(unique_genres)







# 장르 컬럼을 펼쳐서 개별 장르를 포함한 새로운 데이터프레임 생성
expanded_df = df.explode('Genres')

# 장르별로 그룹화하고 평균 평점 계산
genre_avg_scores = expanded_df.groupby('Genres')['IMDB Score'].mean()

# 평균 평점을 기준으로 내림차순 정렬
sorted_genre_avg_scores = genre_avg_scores.sort_values(ascending=False)

# 정렬된 결과 출력
print(sorted_genre_avg_scores.head())








from scipy import stats
import numpy as np

# English 언어인 경우와 그렇지 않은 경우의 평점 추출
english_scores = df[df['Language'] == 'English']['IMDB Score']
non_english_scores = df[df['Language'] != 'English']['IMDB Score']

english_scores.plot(kind="hist")
non_english_scores.plot(kind="hist")

import numpy as np
np.square(english_scores).plot(kind="hist")
np.square(non_english_scores).plot(kind="hist")

# 평균 평점 계산
english_mean = english_scores.mean()
non_english_mean = non_english_scores.mean()

# 스퀘어 변환 적용
english_scores_squared = np.square(english_scores)
non_english_scores_squared = np.square(non_english_scores)

# Levene's test 수행
statistic, p_value = stats.levene(english_scores_squared, non_english_scores_squared)

# 결과 출력
print('Levene Statistic (Squared Data):', statistic)
print('P-Value (Squared Data):', p_value)

# 등분산성 가정이 성립하는 경우
if p_value > 0.05:
    # 등분산성 가정하고 t-test 수행
    t_statistic, t_p_value = stats.ttest_ind(english_scores_squared, non_english_scores_squared, equal_var=True)
    print('Equal Variance T-Test')
    print('T-Statistic (Squared Data):', t_statistic)
    print('P-Value (Squared Data):', t_p_value)
else:
    # 등분산성 가정하지 않고 t-test 수행
    t_statistic, t_p_value = stats.ttest_ind(english_scores_squared, non_english_scores_squared, equal_var=False)
    print('Welch\'s T-Test')
    print('T-Statistic (Squared Data):', t_statistic)
    print('P-Value (Squared Data):', t_p_value)

# 평균 평점 출력
print('English Mean Score:', english_mean)
print('Non-English Mean Score:', non_english_mean)
import matplotlib.pyplot as plt
import seaborn as sns

# 장르 컬럼을 펼쳐서 개별 장르를 포함한 새로운 데이터프레임 생성
expanded_df = df.explode('Genres')

# 장르별로 그룹화하고 영화 개수(count)와 평균 평점(mean) 계산
genre_stats = expanded_df.groupby('Genres')['IMDB Score'].agg(['count', 'mean'])

# 평균 평점을 기준으로 내림차순 정렬
sorted_genre_stats = genre_stats[genre_stats['count'] >= 2].sort_values(by='mean', ascending=False)

# 상위 10개 장르 선택
top_10_genres = sorted_genre_stats.head(10)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='mean', y=top_10_genres.index, data=top_10_genres, palette='viridis')
plt.xlabel('Average IMDB Score')
plt.ylabel('Genres')
plt.title('Top 10 Genres with Highest Average IMDB Scores')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# 장르 컬럼을 펼쳐서 개별 장르를 포함한 새로운 데이터프레임 생성
expanded_df = df.explode('Genres')

# 장르별로 그룹화하고 영화 개수(count)와 평균 평점(mean) 계산
genre_stats = expanded_df.groupby('Genres')['IMDB Score'].agg(['count', 'mean'])

# 카운트가 가장 많은 상위 10개 장르 선택 후 평점을 기준으로 내림차순 정렬
sorted_genre_stats = genre_stats.sort_values(by='count', ascending=False).head(10).sort_values(by='count', ascending=False)



# 시각화
plt.figure(figsize=(12, 8))
ax = sns.barplot(x='mean', y=sorted_genre_stats.index, data=sorted_genre_stats, palette='pastel')

# 눈금에 카운트 수 추가
for i, genre in enumerate(sorted_genre_stats.index):
    count = sorted_genre_stats.loc[genre, 'count']
    ax.text(0.1, i, f'({count})', va='center')

plt.xlabel('Average IMDB Score')
plt.ylabel('Genres')
plt.title('Comparison of Average IMDB Scores among Top 10 Genres with the Highest Counts')
plt.show()



#"Premiere" 열을 datetime 형태로 변환

df["Premiere"] = pd.to_datetime(df["Premiere"])

import numpy as np

friday = df[df["Premiere"].dt.day_name()=="Friday"]["IMDB Score"]
other = df[df["Premiere"].dt.day_name()!="Friday"]["IMDB Score"]

square_friday = np.square(friday)
square_friday.plot(kind="hist")
square_other = np.square(other)
square_other.plot(kind="hist")

from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind

print(shapiro(square_friday))
print(shapiro(square_other))

s,p = levene(square_friday, square_other)
if p > 0.05:
    print(p, "\n등분산성을 만족한다.")
else:
    print(p, "\n등분산성을 만족하지 않는다.")
print("귀무가설은 '개봉 요일과 스코어와의 유의미한 관계가 없다.'이며,\n대립가설은 '개봉 요일과 스코어와의 유의미한 관계가 있다.'이다.")
_, p_value = ttest_ind(square_friday, square_other, alternative="two-sided", equal_var=False)

if p_value > 0.05:
    print(p_value, "귀무가설을 채택한다.")
else:
    print(p_value, "대립가설을 채택한다.")


df['Runtime'].corr(df['IMDB Score'])
df['Runtime'].plot(kind='hist')



from scipy import stats

alpha = 0.05  # 유의수준 설정


# 'Runtime'이 90 이상에서 110인 영화들과 그 외 시간대의 영화들을 나누어 그룹 생성
group_high_runtime = df[(df['Runtime'] >= 90) & (df['Runtime'] <= 110)]
group_other_runtime = df[(df['Runtime'] < 90) | (df['Runtime'] > 110)]

# 각 그룹의 'IMDB Score' 추출
scores_high_runtime = group_high_runtime['IMDB Score']
scores_other_runtime = group_other_runtime['IMDB Score']





# 각 그룹의 'IMDB Score' 시각화

import numpy as np

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
sns.histplot(scores_high_runtime, kde=True)
plt.title('High Runtime Transformed')

plt.subplot(1, 2, 2)
sns.histplot(scores_other_runtime, kde=True)
plt.title('Other Runtime Transformed')

plt.show()






# 등분산성 검정
_, p_value_levene = stats.levene(scores_high_runtime, scores_other_runtime)

if p_value_levene < alpha:
    print("두 그룹은 등분산성을 만족하지 않습니다.")
    t_statistic, p_value_ttest = stats.ttest_ind(scores_high_runtime, scores_other_runtime,equal_var=False)
else:
    print("두 그룹은 등분산성을 만족합니다.")
    t_statistic, p_value_ttest = stats.ttest_ind(scores_high_runtime, scores_other_runtime,equal_var=True)



# one-tailed test
if t_statistic > 0:
    p_value_one_tailed = p_value_ttest / 2
else:
    p_value_one_tailed = 1 - p_value_ttest / 2

if p_value_one_tailed < alpha:
    print("귀무가설을 기각합니다.")
    print("Runtime이 90 이상에서 110인 영화들의 IMDB Score가 그 외 시간대 영화들보다 높다는 통계적으로 유의한 차이가 있습니다.")
else:
    print("귀무가설을 채택합니다.")
    print("Runtime이 90 이상에서 110인 영화들의 IMDB Score와 그 외 시간대 영화들의 IMDB Score 간에는 통계적으로 유의한 차이가 없습니다.")






