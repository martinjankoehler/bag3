import argparse

from bag.core import BagProject
from bag.util.misc import register_pdb_hook

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Write tech info from .tf file.')
    parser.add_argument('tech_name', help='Name of the PDK tech library in Virtuoso.')
    parser.add_argument('out_yaml', help='Name of the output yaml file.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    prj.impl_db.write_tech_info(tech_name=args.tech_name, out_yaml=args.out_yaml)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    run_main(_prj, _args)
