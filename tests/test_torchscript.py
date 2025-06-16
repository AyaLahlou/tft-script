import torch
import os
from tft_torch.tft import TemporalFusionTransformer
import pickle
from omegaconf import OmegaConf, DictConfig
import torch
import tft_torch
print(tft_torch.__file__)
from tft_torch.tft import TemporalFusionTransformer


def test_script_and_save(tmp_path):
    checkpoint= "/glade/u/home/ayal/phenology-ml-clm/models/US_MMS_1ep_nofut_061325.pth"
    filename= "/glade/u/home/ayal/phenology-ml-clm/data/USMMS_to_tft_06132025.pkl"
    configuration = {'optimization':
                    {
                        'batch_size': {'training': 8, 'inference': 8},# both weere 64 before
                        'learning_rate': 1e-4,#was 0.001
                        'max_grad_norm': 1.0,
                    }
                    ,
                    'model':
                    {
                        'dropout': 0.2,#was 0.05 before
                        'state_size': 160,
                        'output_quantiles': [0.1, 0.5, 0.9],
                        'lstm_layers': 4,#was 2
                        'attention_heads': 4 #was 4 #then 6
                    },
                    # these arguments are related to possible extensions of the model class
                    'task_type':'regression',
                    'target_window_start': None, 
                    'checkpoint': checkpoint}

    with open(filename,'rb') as fp:
            data = pickle.load(fp)
        
    feature_map = data['feature_map']
    cardinalities_map = data['categorical_cardinalities']

    structure = {
            'num_historical_numeric': len(feature_map['historical_ts_numeric']),
            'num_historical_categorical': len(feature_map['historical_ts_categorical']),
            'num_static_numeric': len(feature_map['static_feats_numeric']),
            'num_static_categorical': len(feature_map['static_feats_categorical']),
            'num_future_numeric': len(feature_map['future_ts_numeric']),
            'num_future_categorical': len(feature_map['future_ts_categorical']),
            'historical_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['historical_ts_categorical']],
            'static_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['static_feats_categorical']],
            'future_categorical_cardinalities': [cardinalities_map[feat] + 1 for feat in feature_map['future_ts_categorical']],
        }

    configuration['data_props'] = structure




    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    model = TemporalFusionTransformer(config=OmegaConf.create(configuration))
    state_dict = torch.load(checkpoint, map_location=device)
    model_state = model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    
    # 3) save & reload
    out_file = tmp_path / "tft_scripted.pt"
    model.eval()
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, str(out_file))
    assert model_scripted is not None
    #loaded = torch.jit.load(str(out_file))

    # 4) sanity-check: inference runs & shapes match
    #example = torch.randn(2, 36, 10)  # match your forward signature
    #y1 = model(example)
    #y2 = loaded(example)
    #assert y1.shape == y2.shape

if __name__ == "__main__":
    tmp_path = "/glade/u/home/ayal/phenology-ml-clm/models/"
    test_script_and_save(tmp_path)