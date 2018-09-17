
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        #assert(opt.dataset_mode == 'unaligned')
        assert(opt.dataset_mode == 'aligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()

    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
#   assert(opt.dataset_mode == 'unaligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()

    elif opt.model == 'pix2pix_classifier':
        assert(opt.dataset_mode == 'unaligned')
        from .pix2pix_classifier_model import Pix2PixClassifierModel
        model = Pix2PixClassifierModel()

    elif opt.model == 'pix2pix_dunet':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_dunet import Pix2PixDUnet
        model = Pix2PixDUnet()

    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
