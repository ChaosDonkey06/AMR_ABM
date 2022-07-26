from tqdm import tqdm
import numpy as np

def model(patients_state, γ, β, α, movement, ward2size, ward2cluster=None, ρ=6/100):
    """_summary_

    Args:
        patients_state:               _description_
        γ:                            _description_
        β:                            _description_
        α:                            _description_
        movement:                _description_
        ward2size:                    _description_
        ward_oρervation_dict: _description_. Defaults to None.
        ρ:                    Likelihood of positive | test and positive . Defaults to 6/100.

    Returns:
        _type_: _description_
    """

    num_wards    = len(ward2size.values() )
    active_w_df  = movement[["ward_id"]]                                # Active wards
    active_p_df  = movement[["mrn_id", "ward_id", "first_day", "test"]] # Active patients

    num_patients  = patients_state.shape[0]
    num_ensembles = patients_state.shape[1]

    active_wards     = np.unique(active_w_df.values)
    new_patients     = active_p_df[active_p_df.first_day==True]["mrn_id"].values

    γ_ens = γ
    β_ens = β

    if new_patients.shape[0]>0:
        patients_state[new_patients, :]   = np.ones(shape=(new_patients.shape[0], 1)) * γ_ens
        # Potential colonized patients by importation
        prob_patients                    = np.where(patients_state!=0)
        patients_state[prob_patients]    = np.random.random(size=(prob_patients[0].shape)) <= patients_state[prob_patients]


    ward_colonized        = np.full((num_wards, num_ensembles), np.nan)
    ward_nosocomial       = np.full((num_wards, num_ensembles), np.nan)
    ward_imported         = np.full((num_wards, num_ensembles), np.nan)
    ward_positive         = np.full((num_wards, num_ensembles), np.nan)
    ward_negative         = np.full((num_wards, num_ensembles), np.nan)

    num_clusters          = len(set(list(ward2cluster.values())))
    ward_chunk_colonized  = np.full((num_clusters, num_ensembles), np.nan )
    ward_chunk_nosocomial = np.full((num_clusters, num_ensembles), np.nan )
    ward_chunk_imported   = np.full((num_clusters, num_ensembles), np.nan )
    ward_chunk_positive   = np.full((num_clusters, num_ensembles), np.nan )
    ward_chunk_negative   = np.full((num_clusters, num_ensembles), np.nan )

    # compute total imported patients to respective wards
    for _, ward_id in active_wards:
        active_patients_new_ward  = active_p_df[active_p_df.ward_id==ward_id]#["mrn_id"].values
        active_patients_new_ward  = active_patients_new_ward[active_patients_new_ward.first_day==True]["mrn_id"].values

        ward_chunk_imported[ward2cluster[ward_id], :] += np.sum(patients_state[active_patients_new_ward, :], axis=0)
        ward_imported[ward_id, :]                              += np.sum(patients_state[active_patients_new_ward, :], axis=0)

    patients_state    = patients_state.copy()
    imported_patients = patients_state.copy()
    p_status          = patients_state * (1 - np.ones(shape=(num_patients,1)) * α_ens)

    for _, ward_id in enumerate(active_wards):
        active_patients_ward              = active_p_df[active_p_df.ward_id==ward_id]["mrn_id"].values
        C                = np.sum(patients_state[active_patients_ward,:], axis=0)

        λ_ward                          = β_ens  * C / ward2size[ward_id] # Ward force of infection.
        λ                               = λ_ward
        p_status[active_patients_ward, :] = p_status[active_patients_ward, :] + λ

    prob_patients                                = np.where(p_status!=0)
    p_status[prob_patients[0], prob_patients[1]] = np.random.random(size=(prob_patients[0].shape)) <= p_status[prob_patients[0], prob_patients[1]]
    patients_state                               = p_status.copy()

    patients_state_tested     = ρ * patients_state

    patients_state_tested     = np.random.random(size=(num_patients, num_ensembles)) <=  patients_state_tested
    patients_state_not_tested = patients_state-patients_state_tested

    # compute total imported patients to respective wards
    for ward_id in active_wards:
        active_patients_new_ward           = active_p_df[active_p_df.ward_id==ward_id]["mrn_id"].values

        chunk_id = ward2cluster[ward_id]

        ward_chunk_colonized[chunk_id, :]  += np.sum(patients_state[active_patients_new_ward, :], axis=0)
        ward_colonized[ward_id, :]         += np.sum(patients_state[active_patients_new_ward, :], axis=0)

        ward_chunk_nosocomial[chunk_id, :] += np.sum(patients_state[active_patients_new_ward, :]-imported_patients[active_patients_new_ward, :], axis=0)
        ward_nosocomial[ward_id, :]        += np.sum(patients_state[active_patients_new_ward, :]-imported_patients[active_patients_new_ward, :], axis=0)

        active_patients_new_ward           = active_p_df[active_p_df.ward_id==ward_id]
        active_patients_detected_ward      = active_patients_new_ward[active_patients_new_ward.test==True]["mrn_id"].values

        ward_chunk_positive[chunk_id, :]   += np.sum(patients_state_tested[active_patients_detected_ward, :], axis=0)
        ward_positive[ward_id, :]          += np.sum(patients_state_tested[active_patients_detected_ward, :], axis=0)

        ward_chunk_negative[chunk_id, :]   += np.sum(patients_state_not_tested[active_patients_detected_ward, :], axis=0)
        ward_negative[ward_id, :]          += np.sum(patients_state_not_tested[active_patients_detected_ward, :], axis=0)

    return patients_state,  ward_colonized, ward_nosocomial, ward_imported, ward_positive, ward_chunk_colonized, ward_chunk_nosocomial, ward_chunk_imported, ward_chunk_positive, ward_negative, ward_chunk_negative


def model_inference(patients_state, γ, β, α, movement, ward2size, ward2cluster, se=6/100):
    """[summary]

    Args:
        patients_state ([type]): Numpy array with patient state per ensemble. shape: num_patients x num_ensembles.
        wards_state    ([type]): Numpy array with patient state per ensemble. shape: num_wards x num_ensembles.
        param_ens      ([type]): Dictionary with parameters to iterate de model.
        movement  ([type]): DataFrame with movement information.

    Returns:
        patients_state      ([type]): Numpy array with patient state per ensemble after model step. shape: num_patients x num_ensembles.
        wards_state         ([type]): Numpy array with patient state per ensemble after model step. shape: num_wards x num_ensembles.
        colonized           ([type]): Total colonized patient after model step.
        colonized_imported  ([type]): Total colonized patient by importation after model step.
        positive            ([type]): Total colonized detected patients (positive) after model step.
    """

    num_wards     = len(ward2size.values() )
    active_w_df   = movement[["ward_id"]]                                # Active wards
    active_p_df   = movement[["mrn_id", "ward_id", "first_day", "test"]] # Active patients

    num_patients  = patients_state.shape[0]
    num_ensembles = patients_state.shape[1]

    active_wards     = np.unique(active_w_df.values)
    new_patients     = active_p_df[active_p_df.first_day==True]["mrn_id"].values

    γ_ens = γ
    β_ens = β
    α_ens = α

    # Potential colonized patients by importation
    if new_patients.shape[0]>0:
        patients_state[new_patients, :]  = np.ones(shape=(new_patients.shape[0], 1)) * γ_ens
        prob_patients                    = np.where(patients_state!=0)
        patients_state[prob_patients]    = np.random.random(size=(prob_patients[0].shape)) <= patients_state[prob_patients]


    ward_positive         = np.zeros((num_wards, num_ensembles) )
    ward_negative         = np.zeros((num_wards, num_ensembles) )

    num_clusters       = len(set(list(ward2cluster.values())))
    cluster_positive   = np.zeros((num_clusters, num_ensembles) )
    cluster_negative   = np.zeros((num_clusters, num_ensembles) )

    patients_state    = patients_state.copy()

    p_status           = patients_state * (1 - np.ones(shape=(num_patients,1)) * α_ens)

    for idx_ward, ward_id in enumerate(active_wards):
        active_patients_ward              = active_p_df[active_p_df.ward_id==ward_id]["mrn_id"].values
        colonized_patients                = np.sum(patients_state[active_patients_ward,:], axis=0)

        λ_ward                            = β_ens  * colonized_patients / ward2size[ward_id] # Ward force of infection.
        λ                                 = λ_ward
        p_status[active_patients_ward, :] = p_status[active_patients_ward, :] + λ

    prob_patients   = np.where(p_status!=0)

    p_status[prob_patients[0], prob_patients[1]] = np.random.random(size=(prob_patients[0].shape)) <= p_status[prob_patients[0], prob_patients[1]]
    patients_state                               = p_status.copy()


    patients_state_tested = se * patients_state

    patients_state_tested     = np.random.random(size=(num_patients, num_ensembles)) <=  patients_state_tested
    patients_state_not_tested = patients_state-patients_state_tested

    # compute total imported patients to respective wards
    for ward_id in active_wards:
        chunk_id = ward2cluster[ward_id]
        active_patients_new_ward         = active_p_df[active_p_df.ward_id==ward_id]
        active_patients_detected_ward    = active_patients_new_ward[active_patients_new_ward.test==True]["mrn_id"].values

        cluster_positive[chunk_id, :]   += np.sum(patients_state_tested[active_patients_detected_ward, :], axis=0)
        ward_positive[ward_id, :]       += np.sum(patients_state_tested[active_patients_detected_ward, :], axis=0)

        cluster_negative[chunk_id, :]   += np.sum(patients_state_not_tested[active_patients_detected_ward, :], axis=0)
        ward_negative[ward_id, :]       += np.sum(patients_state_not_tested[active_patients_detected_ward, :], axis=0)

    return patients_state, ward_positive, cluster_positive, ward_negative, cluster_negative


def simulate_model(movement_data, ward2size, ward2community, θ, abm_settings, model=model):
    """ Simulate model using point parameters in param_dict.

    Args:
        model          : _description_
        movement_data  : _description_
        ward2size      : _description_
        ward2community : _description_
        θ              : _description_
        abm_settings   : _description_

    Returns:
        _type_: _description_
    """
    # Simulate Model
    α_ens     = np.random.uniform( 1/365, 1/175, size=(abm_settings["num_patients"], abm_settings["num_ensembles"]))
    γ_truth   = θ["γ"]
    β_truth   = θ["β"]

    γ_ens  = np.ones((1, abm_settings["num_ensembles"])) * γ_truth
    β_ens  = np.ones((1, abm_settings["num_ensembles"])) * β_truth

    ward_positive           = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards"], abm_settings["num_ensembles"]), np.nan)
    ward_colonized          = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards"], abm_settings["num_ensembles"]), np.nan)
    ward_nosocomial         = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards"], abm_settings["num_ensembles"]), np.nan)
    ward_colonized_imported = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards"], abm_settings["num_ensembles"]), np.nan)

    cluster_positive           = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards_chunks"], abm_settings["num_ensembles"]), np.nan)
    cluster_colonized          = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards_chunks"], abm_settings["num_ensembles"]), np.nan)
    cluster_nosocomial         = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards_chunks"], abm_settings["num_ensembles"]), np.nan)
    cluster_colonized_imported = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards_chunks"], abm_settings["num_ensembles"]), np.nan)

    ward_negative                 = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards"], abm_settings["num_ensembles"]), np.nan)
    cluster_negative           = np.full((len(abm_settings["dates"] ),  abm_settings["num_wards_chunks"], abm_settings["num_ensembles"]), np.nan)

    patients_state         = np.zeros((abm_settings["num_patients"], abm_settings["num_ensembles"]))

    for i_d, date in tqdm(enumerate(list(abm_settings["dates"] ))):
        movement_date = movement_data.loc[date]

        patients_state, ward_colonized[i_d,:], ward_nosocomial[i_d,:], ward_colonized_imported[i_d,:], ward_positive[i_d,:], cluster_colonized[i_d,:], \
            cluster_nosocomial[i_d], cluster_colonized_imported[i_d,:], cluster_positive[i_d,:], ward_negative[i_d, :], cluster_negative[i_d,:] = model(patients_state, γ_ens, β_ens, α_ens, movement_date, ward2size, ward2community)

    return ward_colonized, ward_nosocomial, ward_colonized_imported, ward_positive, cluster_colonized, cluster_nosocomial, cluster_colonized_imported, cluster_positive, ward_negative, cluster_negative
