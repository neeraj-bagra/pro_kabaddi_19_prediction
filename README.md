# Project Title
Pro Kabaddi League 2019 Predictions


## List of Predictions

1. Predict the winner of the tournament
2. Predict the top team in the points table after the completion of the league matches
3. Predict the team with the highest points for successful raids
4. Predict the team with the highest points for successful tackles
5. Predict the team with the highest super-performance total.
6. Predict the player with the highest SUCCESSFUL RAID percentage.
7. Predict the player with the highest SUCCESSFUL TACKLE percentage


## Data source

All data that we have used in this project is collected from https://www.prokabaddi.com/

Mainly we have collected 3 types of data:
1. Matchwise team performance data for recent seasons (i.e season 5,6,7)
2. Playerwise team performance data for recent seasons (i.e season 5,6,7)
3. Future match schedules

## Cleaning Data
As the data is collected from authorized website of pro kabaddi league, no data is missing.


## Data Preparation

From matchwise team performance data we have derived averages of below points for every team & every series. 
We have used this data as features of a team in a given season:<br>
    - all_out<br>
    - total_raid_points<br>
    - total_tackle_points<br>
    - successful_raids_ratio<br>
    - unsuccessful_raids_ratio<br>
    - successful_tackles_ratio

```
team_agg_feature=df.groupby(['team_id','series_number']).agg({'all_out':nm.mean,
                                                            'total_raid_points':nm.mean,
                                                            'total_tackle_points':nm.mean,
                                                            'successful_raids_ratio':nm.mean,
                                                            'unsuccessful_raids_ratio':nm.mean,
                                                            'successful_tackles_ratio':nm.mean}).reset_index()
```

## Model Building

We have trained model with team features collected (in Data Preparation step) as predicted variables (for self and opponent teams) to predict is_win (dependent variable).

For this we have tried various models like Logistic Regression, SVM,Decision Tree Classifier.

For feature selection we have manually dropped features one by one to make all the features significant (with p_values < 0.05) and independent to each other (with VIF <3). 

Our final model have following features:

|  feature                           |   p_values |    VIF |   coefficient |
| :----------------------------------|-----------:|-------:|--------------:|
|  opponent_successful_tackles_ratio |      0.001 |   1.09 |    -0.0720104 |
|  successful_tackles_ratio          |      0     |   1.08 |     0.116165  |
|  opponent_total_raid_points        |      0     |   1.08 |    -0.0890186 |
|  total_raid_points                 |      0     |   1.07 |     0.0734461 |
|  const                             |      0     | nan    |     0.435101  |


One of the major 'win deciding feature' for any team is successful_tackles_ratio followed by opponent_total_raid_points, total_raid_points, opponent_successful_tackles_ratio with coefficients (0.116,-0.089,0.073,-0.072) respectively.

Logistic Regression, SVM, Decision Tree Classifier models performance was measured using accuracy score.
Model with best accuracy score was Logistic Regression with accuracy 65% and 63% on train and test data set (after hypertuning with C and penalty terms using sklearn library GridSearchCV).

We have used the Logistic Regression Model to predict the winning of all league matches. 
Using previous match statistics we have set cutoff: 
if probability > 51.2874 ( win )
0.487126 <= probability <= 0.512874 ( tie )
probability < 0.403423 ( loss by more than 7 points )
else => loss by less than or equal to 7 points


## Predictions 

**1. Predict the winner of the tournament**
<br>
**2. Predict the top team in the points table after the completion of the league matches**

Considering above model we have predicted points earned by teams in all future league matches and points tables looks like this:

Logistic Regression, SVM,Decision Tree Classifier models performance was measured using accuracy score.

|    team_id | team_name         |   win |   tie |   loss_less_or_equal_7 |   loss_more_than_7 |   Points |
| ----------:|:------------------|------:|------:|-----------------------:|-------------------:|---------:|
|          2 | Dabang Delhi K.C. |    17 |     2 |                      1 |                  2 |       92 |
|          4 | Bengal Warriors   |    15 |     3 |                      4 |                  0 |       88 |
|          1 | Bengaluru Bulls   |    13 |     1 |                      5 |                  3 |       73 |
|         28 | Haryana Steelers  |    12 |     2 |                      4 |                  4 |       70 |
|          5 | U Mumba           |    11 |     2 |                      6 |                  3 |       67 |
|         30 | U.P. Yoddha       |    11 |     2 |                      4 |                  5 |       65 |

So after league match: <br>
**Top team** in the points table will be **Dabang Delhi K.C**<br>
**First eliminator** match will be between **Bengaluru Bulls and U.P. Yoddha** <br>
**Second eliminator** match will be between **Haryana Steelers and U Mumba** <br>


We have predicted winning chance for each play off match with Logistic Regression Model and final result is: 


|    team_id | team_name         |   semifinal_prob |   final_prob |   tournament_win_prob |
| ----------:|:------------------|-----------------:|-------------:|----------------------:|
|          4 | Bengal Warriors   |             1    |         0.62 |                  0.34 |
|          2 | Dabang Delhi K.C. |             1    |         0.58 |                  0.31 |
|          1 | Bengaluru Bulls   |             0.58 |         0.26 |                  0.13 |
|         28 | Haryana Steelers  |             0.51 |         0.2  |                  0.08 |
|          5 | U Mumba           |             0.49 |         0.18 |                  0.08 |
|         30 | U.P. Yoddha       |             0.42 |         0.15 |                  0.06 |


 - semifinal_prob means probability of team reaching semifinal round
 - final_prob means probability of team reaching final round
 - tournament_win_prob means probability of team reaching final round

**Bengal Warriors has highest chances (34% chances) of winning tournament.**


**3. Predict the team with the highest points for successful raids**<br>
**4. Predict the team with the highest points for successful tackles**<br>
**5. Predict the team with the highest super-performance total.**

For 3rd, 4th & 5th predictions, we have first predicted total number of matches played by every team (league match + playoff match) using above logistic regression model and final outcome is:

|  team_name             |   total_match_count |
| :----------------------|--------------------:|
|  Dabang Delhi K.C.     |                  24 |
|  Bengal Warriors       |                  24 |
|  Bengaluru Bulls       |                  24 |
|  Haryana Steelers      |                  24 |
|  Patna Pirates         |                  22 |
|  Telugu Titans         |                  22 |
|  Puneri Paltan         |                  22 |
|  U Mumba               |                  23 |
|  Tamil Thalaivas       |                  22 |
|  Jaipur Pink Panthers  |                  22 |
|  U.P. Yoddha           |                  23 |
|  Gujarat Fortunegiants |                  22 |


Also, we have calculated average number of successful raids, successful tackles and super-performance total per match and then multiplied both and total succesful raids points comes to this:

**Total Raid Points**

|    team_id |   total_match_count |   total_raid_points_sum | team_name             |
| ----------:|--------------------:|------------------------:|:----------------------|
|          2 |                  24 |                 524.4   | Dabang Delhi K.C.     |
|          4 |                  24 |                 511.2   | Bengal Warriors       |
|          1 |                  24 |                 496.421 | Bengaluru Bulls       |
|         28 |                  24 |                 476.211 | Haryana Steelers      |
|          6 |                  22 |                 426.105 | Patna Pirates         |
|          8 |                  22 |                 405.778 | Telugu Titans         |
|          7 |                  22 |                 393.8   | Puneri Paltan         |
|          5 |                  23 |                 393.421 | U Mumba               |
|         29 |                  22 |                 378.4   | Tamil Thalaivas       |
|          3 |                  22 |                 370.7   | Jaipur Pink Panthers  |
|         30 |                  23 |                 369.278 | U.P. Yoddha           |
|         31 |                  22 |                 352     | Gujarat Fortunegiants |

So **highest points for successful raids points will be earned by team Dabang Delhi K.C.**

Similarly, for successful tackles:

**Total Tackles Points**

|    team_id |   total_match_count |   total_tackle_points_sum | team_name             |
| ----------:|--------------------:|--------------------------:|:----------------------|
|          7 |                  22 |                   255.2   | Puneri Paltan         |
|         28 |                  24 |                   246.316 | Haryana Steelers      |
|         30 |                  23 |                   245.333 | U.P. Yoddha           |
|          1 |                  24 |                   241.263 | Bengaluru Bulls       |
|          3 |                  22 |                   240.9   | Jaipur Pink Panthers  |
|          4 |                  24 |                   230.4   | Bengal Warriors       |
|          2 |                  24 |                   224.4   | Dabang Delhi K.C.     |
|         31 |                  22 |                   223.3   | Gujarat Fortunegiants |
|          8 |                  22 |                   222.444 | Telugu Titans         |
|          5 |                  23 |                   221.526 | U Mumba               |
|          6 |                  22 |                   221.158 | Patna Pirates         |
|         29 |                  22 |                   188.1   | Tamil Thalaivas       |

So **highest points for successful tackles will be earned by team Puneri Paltan.**

Similarly, for successful tackles:

**Total Super-Performance Total**

|    team_id |   super_performance_total | team_name             |
| ----------:|--------------------------:|:----------------------|
|          3 |                        41 | Jaipur Pink Panthers  |
|          6 |                        41 | Patna Pirates         |
|          1 |                        38 | Bengaluru Bulls       |
|          8 |                        35 | Telugu Titans         |
|         28 |                        35 | Haryana Steelers      |
|          7 |                        34 | Puneri Paltan         |
|          4 |                        29 | Bengal Warriors       |
|          5 |                        27 | U Mumba               |
|         30 |                        26 | U.P. Yoddha           |
|         31 |                        23 | Gujarat Fortunegiants |
|         29 |                        16 | Tamil Thalaivas       |
|          2 |                        14 | Dabang Delhi K.C.     |

So team with **highest super-performance total will be Jaipur Pink Panthers**


**6. Predict the player with the highest SUCCESSFUL RAID percentage.**<br>
**7. Predict the player with the highest SUCCESSFUL TACKLE percentage**

For successful raid and tackle percentage we have used playerwise data and by using group by function we got following results-

**SUCCESSFUL RAID percentage**

|    player_id |   percentage_success_raid | player_name   | team_name        |   series_number |
| ------------:|--------------------------:|:--------------|:-----------------|----------------:|
|         2514 |                       100 | Chand Singh   | Haryana Steelers |               7 |

**Chand Singh** from **Haryana Steelers** will have **maximum successful raid percentage** with percentage equals to 100%.

**SUCCESSFUL TACKLE percentage**

Similarly **Highest Succesful tackles percentage will be 100%** and below three players are with 100% tackles percentage:
<br>
**1. Lalit chaudhary-from Gujarat Fortunegiants**<br>
**2. Mohit Balyan from U Mumba**<br>
**3. Ankush- from U.P. Yoddha**


## Contributors

1. Neeraj Bagra
2. Aruna Sri Turlapati 
3. Anurag Jain
4. Rohith
