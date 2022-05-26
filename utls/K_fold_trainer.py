def kfold(train_sets,val_sets=None,test_set=None , epochs=1001,device='cpu'):

    ####################################################
    best_val_error_f=[]
    best_model_f=[]
    all_train_errors=[]
    all_validation_errors=[]
    all_test_errors=[]
    for fold in range(5):
        print(f'FOLD {fold}')
        print('--------------------------------')
        train_loader = DataLoader(
                         train_sets[fold], 
                          batch_size=64, shuffle = True,drop_last=True)

        val_loader = DataLoader(
                          val_sets[fold],
                          batch_size=64, shuffle = True , drop_last = True)
        best_val_error = None
        best_model = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_errors, valid_errors,test_errors = [], [],[]
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.95, patience=10,
                                                           min_lr=0.00001)
        for epoch in range(1, epochs):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss, train_error = train(model, train_loader,epoch,device,optimizer,scheduler)
            val_error = test(model, val_loader,device)
            train_errors.append(train_error)
            valid_errors.append(val_error)
            test_er=None
            if(test_set != None):
                test_er = test(model, test_loader,device)
                test_errors.append(test_er)
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error
                best_model = copy.deepcopy(model)

            print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}'
                  .format(epoch, lr, loss, val_error))
        print('leng of test errors = ', len(test_errors))
        all_train_errors.append(train_errors)
        all_validation_errors.append(valid_errors)
        all_test_errors.append(test_errors)
        best_model_f.append(best_model)
        best_val_error_f.append(best_val_error)
    return all_train_errors,all_validation_errors,all_test_errors,best_val_error_f,best_model_f
