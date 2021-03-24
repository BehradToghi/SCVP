#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import logging

from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites import CoRL2017, BasicExperimentSuite
from agent.runnable_model import A3CAgent


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
        metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
        + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the benchmark'
        )
    argparser.add_argument(
        '--corl-2017',
        action='store_true',
        help='If you want to benchmark the corl-2017 instead of the Basic one'
    )
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )


    args = argparser.parse_args()
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    agent = A3CAgent(args.city_name, args_file='agent/trained_model/args.txt',
                     model_file='agent/trained_model/9600000.h5', n_actions=9, frameskip=1)

    # We instantiate an experiment suite. Basically a set of experiments
    # that are going to be evaluated on this benchmark.
    if args.corl_2017:
        experiment_suite = CoRL2017(args.city_name)
    else:
        experiment_suite = BasicExperimentSuite(args.city_name)

    # Now actually run the agent_benchmark
    run_driving_benchmark(agent, experiment_suite, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port)
