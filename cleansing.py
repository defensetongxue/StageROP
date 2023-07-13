from ridgeSegModule import generate_ridge
if __name__=='__main__':
    from config import get_config
    
    args=get_config()
    generate_ridge(args.path_tar)