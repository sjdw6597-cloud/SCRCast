from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom


def data_provider(args, flag):
    dataset_class = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
                     'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
                     'custom': Dataset_Custom}[args.data]
    dataset = dataset_class(args, root_path=args.root_path, data_path=args.data_path,
                            flag=flag, size=[args.seq_len, args.label_len, args.pred_len],
                            target=args.target)
    if len(dataset) == 0:
        raise ValueError(f'{flag} split has no complete forecasting windows')
    if dataset.data_x.shape[1] != args.enc_in:
        raise ValueError(f'Expected {args.enc_in} channels, got {dataset.data_x.shape[1]}')
    # Validation shuffling matches the source training protocol.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=flag != 'test',
                        num_workers=args.num_workers, drop_last=False)
    return dataset, loader
