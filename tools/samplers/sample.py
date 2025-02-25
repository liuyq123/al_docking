import argparse

from sampler_creator import SamplerCreator
from utils.utils import yaml_parser

def sample(config):

    sampler = SamplerCreator(config['sampler']['name'], config['sampler']['score_field']).get_sampler()

    sampler.create_spark_session(config['spark_config'])
    data = sampler.create_dataframe(file_path=config['data']['file_path'], 
                                    prev_samples_path=config['data']['prev_samples_path'], 
                                    id_field=config['data']['id_field'], 
                                    sampling_cutoff=config['data']['sampling_cutoff'])

    samples = sampler.sample(data, **config['sampler']['params'])

    sampler.save_samples(samples, config['data']['output_path'])
    sampler.close_spark_session()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    
    args = parser.parse_args()

    config = yaml_parser(args.config)

    sample(config)

if __name__ == "__main__":
    main()