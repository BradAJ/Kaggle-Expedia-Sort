import pandas as pd
from pandas.io import sql
import MySQLdb
import numpy as np
import sklearn.ensemble
import math
import pprint





def trainer(train_df, col_list, model = None, train_loc1 = 7000014, train_loc2 = 9000007, cv_loc1 = 9000008, cv_loc2 = 9917530, print_factors = True):
    
    #exclude outlying prices for training.
    train_df = train_df.loc[(train_df['price_usd'] <= 2000.0) * (train_df['price_usd'] != 0.0)]
    
    if model is None:
        model = sklearn.ensemble.RandomForestClassifier(n_estimators = 100, min_samples_split = 1000)
    
    model.fit(train_df.loc[train_loc1:train_loc2, col_list], train_df.loc[train_loc1:train_loc2, "booking_bool"])
    
    feature_scores_pairs = [[model.feature_importances_[i], col_list[i]] for i in range(len(col_list))]
    
    if hasattr(model, 'predict_proba'):
        crossval_pred_arr = model.predict_proba(train_df.loc[cv_loc1:cv_loc2, col_list])[:, 1]
    else:
        crossval_pred_arr = model.predict(train_df.loc[cv_loc1:cv_loc2, col_list])

    ndcg = ndcg_calc(train_df.loc[cv_loc1:cv_loc2], crossval_pred_arr)
    
    if print_factors:
        print ndcg
        pprint.pprint(sorted(feature_scores_pairs, reverse = True))
        print model
    
    if return_model:
        return model, feature_scores_pairs
    else:
        return ndcg

 
def ndcg_calc(train_df, pred_scores):
    """
    Calculate Normalized Discounted Cumulative Gain for a dataset is ranked with pred_scores (higher score = higher rank).
    If 'booking_bool' == 1 then that result gets 5 points.  If 'click_bool' == 1 then that result gets 1 point (except:
    'booking_bool' = 1 implies 'click_bool' = 1, so only award 5 points total).  
    
    DCG = Sum( (2 ** points - 1) / log2(rank_in_results + 1) )
    The numerator is 31 for a booking and 1 for a click.
    IDCG = Maximum possible DCG given the set of bookings/clicks in the training sample.
    
    """
    eval_df = train_df[['srch_id', 'booking_bool', 'click_bool']]
    eval_df['score'] = pred_scores

    logger = lambda x: math.log(x + 1, 2)
    eval_df['log_rank'] = eval_df.groupby(by = 'srch_id')['score'].rank(ascending = False).map(logger)

    book_dcg = (eval_df['booking_bool'] * 31.0 / eval_df['log_rank']).sum()
    book_idcg = (31.0 * eval_df['booking_bool']).sum()
    
    click_dcg = (eval_df['click_bool'] * (eval_df['booking_bool'] == 0) / eval_df['log_rank']).sum()
    
    # Max number of clicks in training set is 30.
    # Calculate the 31 different contributions to IDCG that 0 to 30 clicks have
    # and put in dict {num of click: IDCG value}.
    disc = [1.0 / math.log(i + 1, 2) if i != 0 else 0 for i in range(31)]
    disc_dict = { i: np.array(disc).cumsum()[i] for i in range(31)}
    
    # Map the number of clicks to its IDCG and subtract off any clicks due to bookings
    # since these were accounted for in book_idcg.
    click_idcg = (eval_df.groupby(by = 'srch_id')['click_bool'].sum().map(disc_dict) - eval_df.groupby(by = 'srch_id')['booking_bool'].sum()).sum()

    return (book_dcg + click_dcg) / (book_idcg + click_idcg)




def df_from_query(training = True):
    
    dbcon = MySQLdb.connect('localhost', 'kaggler', 'expediapass', 'expedia')

    main_cols = "SELECT row_num, srch_id, s.prop_id, price_usd, prop_location_score1, prop_location_score2, prop_log_historical_price, prop_review_score, random_bool, srch_adults_count, srch_booking_window, srch_children_count, srch_length_of_stay, srch_query_affinity_score, srch_room_count, visitor_hist_adr_usd, (prop_starrating / price_usd) AS star_by_price, (prop_review_score / price_usd) AS rev_by_price"

    agg_cols = ", prop_cnt, promotion_cnt, train_price_avg, (booking_cnt / prop_cnt) AS book_per_pcnt, (promotion_flag / promotion_cnt) AS promo_per_procnt, (train_price_avg / price_usd) AS avg_by_price, ((click_cnt - booking_cnt) / prop_cnt) AS click_nobook_per_pcnt"

    agg_missing = ", 0 AS prop_cnt, 0 AS promotion_cnt, 0 AS train_price_avg, 0 AS book_per_pcnt, 0 AS promo_per_procnt, 0 AS avg_by_price, 0 AS click_nobook_per_pcnt"

    if training:
        from_tables1 = " FROM TrainSearch AS s, PropFactors7M AS p"
        from_tables2 = " FROM TrainSearch AS s"
    else:
        from_tables1 = " FROM TestSearch AS s, PropFactors AS p"
        from_tables2 = " FROM TestSearch AS s"

    train_dfmost = sql.frame_query(main_cols + agg_cols + from_tables1 + " WHERE p.prop_id = s.prop_id AND srch_id >=" + str(srch_start) + " AND srch_id <= "+ str(srch_end)+";", con = dbcon, index_col = 'row_num')

    train_dfpropmissing = sql.frame_query(main_cols + agg_missing + from_tables2 + " WHERE prop_id NOT IN (SELECT prop_id FROM PropFactors) AND srch_id >=" + str(srch_start) + " AND srch_id <= "+ str(srch_end)+";", con = dbcon, index_col = 'row_num')

    dbcon.close()

    train_df = pd.concat([train_dfmost, train_dfpropmissing])
    train_df.sort(inplace = True)

    train_df_agg = agg_by_srch(train_df)
    train_df_agg.fillna(value = 0, inplace = True)

    return train_df_agg



def agg_by_srch(train_df):
    cols_in = ['prop_location_score1', 
               'prop_location_score2', 
               'book_per_pcnt', 
               'avg_by_price', 
               'rev_by_price', 
               'star_by_price', 
               'promo_per_procnt',
               'prop_log_historical_price', 
               'click_nobook_per_pcnt', 
               'ploc_score1_by_price', 
               'ploc_score2_by_price']

    cols_out = ['srch_ploc_score1_mean', 
                'srch_ploc_score2_mean', 
                'srch_book_per_pcnt_mean', 
                'srch_avg_by_price_mean', 
                'srch_rev_by_price_mean', 
                'srch_star_by_price_mean', 
                'srch_promo_per_procnt_mean', 
                'srch_prop_log_historical_price_mean', 
                'click_nobook_per_pcnt_mean', 
                'ploc_score1_mean', 
                'ploc_score2_mean']

    srch_df = train_df[['srch_id'] + cols_in].groupby(by='srch_id').mean()
    srch_df.columns = cols_out

    srch_df['srch_price_med'] = train_df[['srch_id','price_usd']].groupby(by = 'srch_id').median()
    srch_df['srch_id'] = srch_df.index
    
    cols_inp = list(cols_in)
    cols_outp = list(cols_out)
    cols_inp.append('price_usd')
    cols_outp.append('srch_price_med')

    cols_agg = ['ploc_score1_by_mean', 
                'ploc_score2_by_mean', 
                'book_per_pcnt_by_mean', 
                'avg_by_price_by_mean', 
                'rev_by_price_by_mean', 
                'star_by_price_by_mean', 
                'promo_perprocnt_by_mean', 
                'loghistp_by_mean', 
                'click_nobper_pcnt_by_mean', 
                'ploc_score1_by_price_by_mean', 
                'ploc_score2_by_price_by_mean', 
                'price_by_med']

    for i in xrange(len(cols_inp)):
        train_df[cols_agg[i]] = train_df[cols_inp[i]] / train_df.join(srch_df[['srch_id', cols_outp[i]]], on='srch_id', rsuffix = '_agg')[cols_outp[i]]
    
    train_df.fillna(value = 0, inplace = True)
    train_df.loc[train_df.price_by_med == np.inf, 'price_by_med'] = 0

    
    return train_df


def test_pred_sorted(test_df, model, cols, regress = False):
    """
    To save output use: test_out_df.to_csv("FILE OUT", index = False, cols = ['srch_id', 'prop_id'], header = ['SearchId','PropertyId'])
    """
    if regress:
        scores = model.predict(test_df[cols])
    else:
        scores = model.predict_proba(test_df[cols])[:, 1]
    test_df['sort_score'] = scores
     
    return test_df[['srch_id', 'prop_id', 'sort_score']].sort(columns=['srch_id', 'sort_score'], ascending = [True, False])


def loader_saver(model, cols, file_str = "test_preds_saved", regress_model = False):
    """
    Trying to run the MySQL query to load the entire Test set was too memory intensive for my 4Gb machine, 
    so load and process the data in chunks.

    Then join the files together on the command line:
    cat test_preds_saved0.csv test_preds_saved1.csv test_preds_saved2.csv test_preds_saved3.csv test_preds_saved4.csv test_preds_saved5.csv > test_preds_all_ADDHEADER.csv
    """
    
    starts = [2, 110001, 220001, 330001, 440001, 550001]
    ends = [110000, 220000, 330000, 440000, 550000, 700000]

    for i in xrange(6):
        test_df = test_df_out(cols, srch_start = starts[i], srch_end = ends[i])
        
        if regress_model:
            pred_df = test_pred_sorted(test_df, model, cols, regress = True)
        else:
            pred_df = test_pred_sorted(test_df, model, cols, regress = False)
        pred_df.to_csv(file_str + str(i) + ".csv", index = False, cols = ['srch_id', 'prop_id'], header = False)
