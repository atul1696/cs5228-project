import numpy as np
import math

def get_recommendation_weights(row, X, feature_weightage):
    similarity_dict = {}

    ## Are they in the same subzone?
    similarity_dict['subzone'] = np.array(X['subzone']==row['subzone'])

    ## Are they in the same planning area? (Should probably be less importance than subzone)
    similarity_dict['planning_area'] = 0.5*np.array(X['planning_area']==row['planning_area'])
    ## TODO : Maybe we can remove both subzone and planning area and replace it all with lat, lng distance

    ## Are they in the same price bracket?
    price_distance_sq = (X['price'] - row['price'])**2
    price_sigma = 1e7
    similarity_dict['price'] = np.exp(-price_distance_sq/(2*(price_sigma**2)))

    ## Do they have the same property type?
    similarity_dict['property_type'] = np.array(X['property_type']==row['property_type'])

    ## Do they have the same number of beds?
    similarity_dict['num_beds'] = np.array(X['num_beds']==row['num_beds'])
    ## TODO : What about properties with higher num_beds? Does adding them helps?

    ## Are they in the same size bracket?
    size_sqft_distance_sq = (X['size_sqft'] - row['size_sqft'])**2
    size_sqft_sigma = 1e3
    similarity_dict['size_sqft'] = np.exp(-size_sqft_distance_sq/(2*(size_sqft_sigma**2)))
    ## TODO : Add another feature for 'price per square foot'

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
    tenure_left_X = X['tenure_duration'].astype(float) - (2022 - X['built_year'].astype(float))
    tenure_left_row = float(row['tenure_duration']) - (2022 - float(row['built_year']))
    if isinstance(tenure_left_row, float) and math.isnan(tenure_left_row):
        similarity_dict['tenure_left'] = 0
    else:
        tenure_left_distance_sq = (tenure_left_X - tenure_left_row)**2
        tenure_left_sigma = 1e3
        similarity_dict['tenure_left'] = np.exp(-tenure_left_distance_sq/(2*(tenure_left_sigma**2)))
        similarity_dict['tenure_left'] = np.nan_to_num(similarity_dict['tenure_left'], nan=0)

    weights = np.zeros(len(X))
    for ele in similarity_dict:
        weights = weights + feature_weightage[ele]*similarity_dict[ele]

    return weights
