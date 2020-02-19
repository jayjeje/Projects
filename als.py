import sys
import pandas as pd
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import random
from google.cloud import bigquery
from google.oauth2 import service_account
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
import implicit
from sklearn.preprocessing import MultiLabelBinarizer

credential_path = 'bonnie/mount/Tailor Project-2891a87e1dd9.json'

def get_like_pouch_df(credential_path):

    project_id = 'tailor-project-216201'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=project_id)

    query = """
        WITH like_filtering AS (
          SELECT
            fullVisitorId,
            clientId,
            eventAction,
            eventLabel AS liked_item, # 좋아요 또는 좋아요 취소 제품
            export_datetime_kr,
            row_number() over(partition by fullVisitorId, clientId, eventLabel order by export_datetime_kr asc) as row  

          FROM
            `tailor-project-216201.temp_gh.realtime_ga_session_query`
          WHERE
            eventAction IN ('좋아요', '좋아요 취소') # 좋아요, 좋아요 취소 함께 시간 순 정렬
            and eventLabel is not null
          ORDER BY
            fullVisitorId,
            eventLabel,
            export_datetime_kr
        ),
        like_filtering2 as(
        select
          fullVisitorId,
          clientId,
          liked_item,
          max(row) max_row
        from
          like_filtering
        group by
          fullVisitorId,
          clientId,
          liked_item
        ),
        like_filter_final as(
        select
          a.*, 
          b.max_row
        from
          like_filtering a
        left outer join
          like_filtering2 b 
          on a.fullVisitorId = b.fullVisitorId 
          and a.clientId = b.clientId 
          and a.liked_item = b.liked_item
        ),
        like_records_raw AS (
        SELECT
            fullVisitorId,
            clientId,
            eventAction,
            SUBSTR(liked_item, -10) new_product_code10,
            SUBSTR(liked_item, -9) new_product_code9,
            liked_item,
            export_datetime_kr
        from
          like_filter_final
        where
          row = max_row
          and eventAction != '좋아요 취소'
          ), 

        pd as(
        select
          product_code
        from `tailor-project-216201.temp_gh.product_data`
        ),

        like_records AS (
        SELECT
            fullVisitorId,
            clientId,
            eventAction,
            split(liked_item, '+')[safe_offset(1)] AS item,
            export_datetime_kr
        FROM
          like_records_raw
        WHERE
          new_product_code9 in (select product_code from pd) 
          or new_product_code10 in (select product_code from pd)
          ), 

        pouch_filtering AS (
          SELECT
            fullVisitorId,
            clientId,
            eventAction,
            eventLabel AS in_pouch_item,
            export_datetime_kr,
            row_number() over(partition by fullVisitorId, clientId, eventLabel order by export_datetime_kr asc) as row  

          FROM
            `tailor-project-216201.temp_gh.realtime_ga_session_query`
          WHERE
            eventAction IN ('파우치 담기', '파우치 담기취소')
            and eventLabel is not null
          ORDER BY
            fullVisitorId,
            eventLabel,
            export_datetime_kr
        ),
        pouch_filtering2 as(
        select
          fullVisitorId,
          clientId,
          in_pouch_item,
          max(row) max_row
        from
          pouch_filtering
        group by
          fullVisitorId,
          clientId,
          in_pouch_item
        ),
        pouch_filter_final as(
        select
          a.*, 
          b.max_row
        from
          pouch_filtering a
        left outer join
          pouch_filtering2 b 
          on a.fullVisitorId = b.fullVisitorId 
          and a.clientId = b.clientId 
          and a.in_pouch_item = b.in_pouch_item
        ),


        in_pouch_records_raw AS (
          SELECT
            fullVisitorId,
            clientId,
            eventAction,
            SUBSTR(in_pouch_item, -10) new_product_code10,
            SUBSTR(in_pouch_item, -9) new_product_code9,
            in_pouch_item,
            export_datetime_kr
        from
          pouch_filter_final
        where
          row = max_row
          and eventAction != '파우치 담기취소'
            ),

        in_pouch_records AS (
          SELECT
            fullVisitorId,
            clientId,
            eventAction,
            split(in_pouch_item, '+')[safe_offset(1)] AS item,
            export_datetime_kr
          FROM
            in_pouch_records_raw
          WHERE
            new_product_code9 in (select product_code from pd) 
            or new_product_code10 in (select product_code from pd)
            )
        SELECT
          *
        FROM 
          like_records
        UNION ALL
        SELECT
          *
        FROM 
          in_pouch_records
        ORDER BY
          fullVisitorId, export_datetime_kr
        """

    df = client.query(query, project=project_id).to_dataframe()

    return df


def get_product_df(credential_path):

    project_id = 'tailor-project-216201'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=project_id)

    query = """
        select
            *
        from `tailor-project-216201.temp_gh.product_data`
         """

    df = client.query(query, project=project_id).to_dataframe()

    return df


def get_review_df(credential_path):


    project_id = 'tailor-project-216201'
    credentials = service_account.Credentials.from_service_account_file(credential_path)
    client = bigquery.Client(credentials=credentials, project=project_id)

    query = """
       With pd as(
    select
      product_code
    from `tailor-project-216201.temp_gh.product_data`
    )


    Select
          fullVisitorId,
          screenName,
          '리뷰조회' as eventAction,
          CASE WHEN eventCategory like '%리뷰 조회' and eventAction is not null and eventAction != '리뷰 조회' THEN split(eventAction, '+')[safe_offset(1)]
                   WHEN (eventAction = '상품 리뷰 더보기' or (eventCategory ='리뷰 조회' and eventAction = '리뷰 조회')) and eventLabel is not null THEN split(eventLabel, '+')[safe_offset(1)]
          END item,
          visit_datetime_kr,
          export_datetime_kr

        From 
          `tailor-project-216201.temp_gh.realtime_ga_session_query`

        Where

          eventCategory like '%리뷰 조회'
          AND
          eventAction IS NOT NULL
          and
          eventLabel != '본인'
          AND
          (split(eventAction, '+')[safe_offset(1)] in (select product_code from pd) or split(eventLabel, '+')[safe_offset(1)] in (select product_code from pd))
          or
          eventAction =  '상품 리뷰 더보기'
          

        order by 
                fullVisitorId, export_datetime_kr, screenName
        """

    df = client.query(query, project=project_id).to_dataframe()


    return df



def get_item_dict_build(product_df, product_code):
    item_lst = list(product_df[product_code].unique())
    item_lst.sort()
    item_dict_to_build = dict(zip(item_lst, range(0, len(item_lst))))
    return item_dict_to_build


def get_item_dict_map(product_df, product_code):
    item_lst = list(product_df[product_code].unique())
    item_lst.sort()
    item_dict_to_map = dict(zip(range(0, len(item_lst)), item_lst))
    return item_dict_to_map


def build_user_action_profile(log_df, product_df, id, item, action):
    # 데이터 프레임 (액션별 추출), 아이템 : 숫자 딕셔너리
    temp_df = log_df[log_df.eventAction.isin(action)].replace({item: get_item_dict_build(product_df, 'product_code')})
    temp_series = temp_df.groupby([id])[item].agg(lambda x: list(x))
    temp_df = pd.DataFrame(temp_series).reset_index()
    return temp_df


def binary_vec(df, id, item):
    mlb = MultiLabelBinarizer()
    temp_vec = pd.DataFrame(mlb.fit_transform(df[item]), columns=mlb.classes_, index=df[id])
    return temp_vec


def vec_frame(df_list, product_df, id):
    longest = 0
    try:
        for each in df_list:
            if len(each) > longest:
                longest = len(each)
                df = each
            else:
                pass
        print(longest, "of", id, "exists in", df)
    except (ValueError, IndexError, RuntimeError, NameError):
        longest = len(view_vec)
    pd_cnt = len(product_df)
    id_cnt = len(df)
    temp_df = pd.DataFrame(np.zeros((id_cnt, pd_cnt)))
    temp_df.index = df.index
    return temp_df


def make_train(ratings, pct_test = 0.2):
    test_set = ratings.copy()  
    test_set[test_set != 0] = 1 # binary preference matrix 로 저장
    
    training_set = ratings.copy()  
    
    nonzero_inds = training_set.nonzero() # action이 일어난 index 파악
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) #  item,user index 합치기 
    
    random.seed(0) 
    
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # 정수로 변환 위해 올림. 
    samples = random.sample(nonzero_pairs, num_samples) # 원하는 수 만큼 랜덤하게 고른다

    item_inds = [index[0] for index in samples] # 아이템 인덱스 

    user_inds = [index[1] for index in samples] # 유저 인덱스

    
    training_set[item_inds, user_inds] = 0 #랜덤하게 선택된 유저-아이템 페어에 0 대입 
    training_set.eliminate_zeros() # 메모리를 위해 0으로 된 것은 제외 (sparse matirx형태에서 압축)
    
    return training_set, test_set, list(set(user_inds))

def build_model():
    
    #load data
    start_time = datetime.datetime.now()
    df = get_like_pouch_df(credential_path)
    product_df = get_product_df(credential_path)
    review_df = get_review_df(credential_path)
    review_df=review_df.dropna()
    end_time = datetime.datetime.now()
    elapsed = round((end_time - start_time).total_seconds(), 2)
    print(f'time elapsed {elapsed} seconds loading data')
    
    
    # simple config
    like_act = ['좋아요']
    pouch_act = ['파우치 담기']
    review_act = ['리뷰조회']


    # process into vector
    like_df = build_user_action_profile(df, product_df, 'fullVisitorId', 'item', like_act)
    pouch_df = build_user_action_profile(df, product_df, 'fullVisitorId', 'item', pouch_act)
    review_agg_df = build_user_action_profile(review_df, product_df, 'fullVisitorId', 'item', review_act)


    like_vec = binary_vec(like_df, 'fullVisitorId', 'item')
    pouch_vec = binary_vec(pouch_df, 'fullVisitorId', 'item')
    review_vec = binary_vec(review_agg_df, 'fullVisitorId', 'item')


    item_dict_map = get_item_dict_map(product_df, 'product_code')

    item_dict_build = get_item_dict_build(product_df, 'product_code')


    # adding vectors to the frame
    df_list = [like_vec, pouch_vec, review_vec]
    empty_vec = vec_frame(df_list, product_df, 'fullVisitorId')


    like_vec = empty_vec.add(like_vec, fill_value=0).sort_index()
    pouch_vec = empty_vec.add(pouch_vec, fill_value=0).sort_index()
    review_vec = empty_vec.add(review_vec, fill_value=0).sort_index()
    
    db=product_df
    
    
    high=[]
    pouch_vec['count']=0
    pouch_vec['count']=pouch_vec.sum(axis=1)

    high = pouch_vec[pouch_vec['count']>4].index.tolist()

    like_vec=like_vec.loc[high, :]
    pouch_vec=pouch_vec.loc[high, :]
    review_vec=review_vec.loc[high, :]
    ##여기까지해서 조건에 맞는 매트릭스만 뽑아진 상태

    #가중치 부여
    like_vec=like_vec*6
    pouch_vec=pouch_vec*2
    review_vec=review_vec*2

    tdf=like_vec.add(pouch_vec,fill_value=0).reset_index()
    tdf=tdf.fillna(0)
    tdf=tdf.set_index('fullVisitorId')
    tdf=tdf.add(review_vec,fill_value=0).reset_index()
    del tdf['count']
    tdf=tdf.set_index('fullVisitorId')

    df2=tdf.transpose()
    sparse_item_user=sparse.csr_matrix(df2)
    sparse_user_item=sparse.csr_matrix(tdf)

    matrix_size = sparse_user_item.shape[0]*sparse_user_item.shape[1] # Number of possible interactions in the matrix
    num_purchases = len(sparse_user_item.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_purchases/matrix_size))
    
    # train, test set 구분
    product_train, product_test, product_users_altered = make_train(sparse_item_user, pct_test = 0.05)


    # 모델 setting
    model = implicit.als.AlternatingLeastSquares(factors=6, regularization=0.1, iterations=40)

    

    alpha_val = 20
    data_conf = (product_train * alpha_val).astype('double') #confidence

    
    model.fit(data_conf) 

    item_vecs = model.item_factors
    user_vecs = model.user_factors
    
    return model, sparse_user_item, high


def recommend(product_df, person_id, sparse_person_content, person_vecs, content_vecs, num_contents=10):
    
    person_interactions = sparse_person_content[person_id,:].toarray() #해당 유저가 보인 인터랙션 다 가져옴
    person_interactions = person_interactions.reshape(-1) + 1 #해당유저 인터랙션에 +1에서 액션 없던거 1로 만듦
    person_interactions[person_interactions > 1] = 0 #1 초과인 원래 인터랙션 보인 아이템은 인터랙션 0으로 만듦
    
    rec_vector = person_vecs[person_id,:].dot(content_vecs.T).toarray() #여기서 해당 사람이 item에 보일 점수 계산 
    
    
    min_max = MinMaxScaler() 
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0] #아이템에 가지는 인터랙션 0,1사이로 스케일링
    
    recommend_vector = person_interactions * rec_vector_scaled #이미 action한거는 제외하기
    
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents] #높은 점수부터 보여주기, 정해진 갯수만큼 
    
    # 번호, 이름, 점수 저장할 리스트들 생성
    code=[]
    titles = []
    scores = []
    db = product_df
    db['description']=db['brand_name']+db['product_name']
    
    item_dict_map = get_item_dict_map(product_df, 'product_code')
    for idx in content_idx:
        # 리스트들에 각 항목들 추가
        code.append(db.loc[db['product_code'] == item_dict_map[idx], 'product_code'].item())
        titles.append(db.loc[db['product_code'] == item_dict_map[idx], 'description'].item())
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'code':code,'title': titles, 'score': scores})

    return pd.DataFrame(recommendations) #추천리스트 항목 반환


def als_recommend(product_df, model, high, sparse_user_item, person_id):
    person_index=high.index(person_id)
    person_vecs = sparse.csr_matrix(model.user_factors)
    content_vecs = sparse.csr_matrix(model.item_factors)
    recommendations = recommend(product_df, person_index, sparse_user_item, person_vecs, content_vecs)
    
    return recommendations