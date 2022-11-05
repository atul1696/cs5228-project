import numpy as np
import math

def get_recommendation_weights(row, X, feature_weightage):
    similarity_dict = {}

    ## Are they in the same subzone?
    similarity_dict['subzone'] = np.array(X['subzone']==row['subzone'])

    ## Are they in the same planning area? (Should probably be less importance than subzone)
    similarity_dict['planning_area'] = 0.5*np.array(X['planning_area']==row['planning_area'])

    ## Are they in the same price bracket?
    price_distance_sq = (X['price'] - row['price'])**2
    price_sigma = 1e7
    similarity_dict['price'] = np.exp(-price_distance_sq/(2*(price_sigma**2)))

    ## Do they have the same property type?
    similarity_dict['property_type'] = np.array(X['property_type']==row['property_type'])

    ## Do they have the same or more number of beds?
    if isinstance(row['num_beds'], float) and math.isnan(row['num_beds']):
        similarity_dict['num_beds'] = 0
    else:
        num_beds_distance = np.nan_to_num(X['num_beds']) - row['num_beds']
        num_beds_distance_sq = num_beds_distance**2
        num_beds_sigma = 2
        num_beds_score = np.exp(-num_beds_distance_sq/(2*(num_beds_sigma**2)))
        similarity_dict['num_beds'] = np.select(condlist=[num_beds_distance >= 0, num_beds_distance<0], choicelist=[num_beds_score, num_beds_score-1])

    ## Do they have the same baths bracket?
    if isinstance(row['num_baths'], float) and math.isnan(row['num_baths']):
        similarity_dict['num_baths'] = 0
    else:
        num_baths_distance_sq = (np.nan_to_num(X['num_baths']) - row['num_baths'])**2
        num_baths_sigma = 1
        similarity_dict['num_baths'] = np.exp(-num_baths_distance_sq/(2*(num_baths_sigma**2)))

    ## Are they in the same price per size sqft bracket?
    price_per_sqft_distance_sq = (X['price_per_sqft'] - row['price_per_sqft'])**2
    price_per_sqft_sigma = 1e3
    similarity_dict['price_per_sqft'] = np.exp(-price_per_sqft_distance_sq/(2*(price_per_sqft_sigma**2)))

    ## Do they have the same floor level?
    if isinstance(row['floor_level'], float) and math.isnan(row['floor_level']):
        similarity_dict['floor_level'] = 0
    else:
        similarity_dict['floor_level'] = np.array(X['floor_level']==row['floor_level'])

    ## Do they have the same level of furnishing?
    if row['furnishing']=='unspecified' :
        similarity_dict['furnishing'] = 0
    else:
        similarity_dict['furnishing'] = np.array(X['furnishing']==row['furnishing'])

    ## How much time is left in the tenure?
    if isinstance(row['tenure_left'], float) and math.isnan(row['tenure_left']):
        similarity_dict['tenure_left'] = 0
    else:
        tenure_left_distance_sq = (np.nan_to_num(X['tenure_left']) - row['tenure_left'])**2
        tenure_left_sigma = 1e3
        similarity_dict['tenure_left'] = np.exp(-tenure_left_distance_sq/(2*(tenure_left_sigma**2)))

    weights = np.zeros(len(X))
    for ele in similarity_dict:
        weights = weights + feature_weightage.get(ele, 0) * similarity_dict[ele]

    return weights
