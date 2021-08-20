"""OmicLearn main file."""
import warnings
import pandas as pd
from PIL import Image
import streamlit as st
from datetime import datetime
warnings.simplefilter("ignore", FutureWarning)

# UI components and others func.
from utils.ui_helper import (main_components,
                             load_data, main_text_and_data_upload, objdict,
                             )

# Set the configs
APP_TITLE = "Predict SEQ Performance"
PARAM_FILE = "model_params.json"

# Main Function
def PredictSeqMain():

    # Define state
    state = objdict()
    state['df'] = pd.DataFrame()

    # Main components
    main_components()
    # Welcome text and Data uploading
    main_text_and_data_upload(state, APP_TITLE, PARAM_FILE)

# Run the OmicLearn
if __name__ == '__main__':
    try:
        PredictSeqMain()
    except (ValueError, IndexError) as val_ind_error:
        st.error(f"There is a problem with values/parameters or dataset due to {val_ind_error}.")
    except TypeError as e:
        # st.warning("TypeError exists in {}".format(e))
        pass
