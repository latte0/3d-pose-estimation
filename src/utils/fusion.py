

class Dataset:
    def __init(self)__:

        self.dataset = {h36m = {}, mpii = {}, mpi_inf_3dhp = {}}

        self.h36mImgSize = 224
        self.nJoints = 16
        self.accIdxs = {1,2,3,4,5,6,11,12,15,16}
        self.flipRef = {{1,6},   {2,5},   {3,4},
                        {11,16}, {12,15}, {13,14}}

        # Pairs of joints for drawing skeleton

        self.skeletonRef = {{1,2,1},    {2,3,1},    {3,7,1},
                            {4,5,2},    {4,7,2},    {5,6,2},
                            {7,9,0},    {9,10,0},
                            {13,9,3},   {11,12,3},  {12,13,3},
                            {14,9,4},   {14,15,4},  {15,16,4}}
        self.nBones = #self.skeletonRef

        self.mpiinf3dhpImgSize = 368
        self.mpiinf3dhgIdx = {11, 10, 9, 12, 13, 14, 15, 16, 2, 1, 5, 4, 3, 6, 7, 8}
        self.mpi_inf_3dhp_num_joint = 14

        -- Index reference
        opt.idxRef = {h36m = {}, mpii = {}, mpi_inf_3dhp = {}}
        opt.testIters = {h36m = {}, mpii = {}, mpi_inf_3dhp = {}}
        opt.validIters = {h36m = {}, mpii = {}, mpi_inf_3dhp = {}}

        opt.testBatch = 1

        annot_3d = {}
        tags_3d = {'action','bbox','camera','id','joint_2d','joint_3d_mono',
                      'subaction','subject','istrain'}

        h5path = ''

        if opt.h36mFullTest:
            h5path = 'data/h36m/annotFullTest.h5'
        else:
            h5path = 'data/h36m/annotSampleTest.h5'

        a_3d = hdf5.open(paths.concat(projectDir, h5path),'r')

        for _,tag in ipairs(tags_3d) do annot_3d[tag] = a_3d:read(tag):all() end

        a_3d:close()

        tp = 'h36m'
        if not opt.idxRef3D:
            allIdxs = torch.range(1,annot_3d.id:size(1))
            self.dataset['h36m'].allSamples = annot_3d.id:size(1)
            opt.idxRef[tp] = {}
            opt.idxRef[tp].test = allIdxs[annot_3d.istrain:eq(0)]
            opt.idxRef[tp].train = allIdxs[annot_3d.istrain:eq(1)]

            -- Set up training/validation split
            perm = torch.randperm(opt.idxRef[tp].train:size(1)):long()
            opt.idxRef[tp].valid = opt.idxRef[tp].test
            opt.idxRef[tp].train = opt.idxRef[tp].train
            opt.nValidImgs = opt.idxRef[tp].valid:size(1)

            opt.idxRef3D = opt.idxRef[tp]

            torch.save(opt.save .. '/options.t7', opt)


        self.dataset[tp].annot = annot_3d
        self.dataset[tp].nsamples = {train=opt.idxRef[tp].train:numel(),
                         valid=opt.idxRef[tp].valid:numel(),
                         test=opt.idxRef[tp].test:numel()}
        opt.testIters[tp] = self.dataset[tp].nsamples.test
        opt.validIters[tp] = self.dataset[tp].nsamples.valid


        annot_2d = {}
        tags_2d = {'index','person','imgname','part','center','scale',
                      'normalize','torsoangle','visible','multi','istrain'}
        a_2d = hdf5.open(paths.concat(projectDir,'data/mpii/annot.h5'),'r')
        for _,tag in ipairs(tags_2d) do annot_2d[tag] = a_2d:read(tag):all() end
        a_2d:close()
        annot_2d.index:add(1)
        annot_2d.person:add(1)
        annot_2d.part:add(1)

        -- Index reference
        tp = 'mpii'
        if not opt.idxRef2D then
            allIdxs = torch.range(1,annot_2d.index:size(1))
            self.dataset[tp].allSamples = annot_2d.index:size(1)
            opt.idxRef[tp] = {}
            opt.idxRef[tp].test = allIdxs[annot_2d.istrain:eq(0)]
            opt.idxRef[tp].train = allIdxs[annot_2d.istrain:eq(1)]

            tmpAnnot = annot_2d.index:cat(annot_2d.person, 2):long()
            tmpAnnot:add(-1)

            validAnnot = hdf5.open(paths.concat(projectDir, 'data/mpii/annot/valid.h5'),'r')
            tmpValid = validAnnot:read('index'):all():cat(validAnnot:read('person'):all(),2):long()
            opt.idxRef[tp].valid = torch.zeros(tmpValid:size(1))
            opt.nValidImgs = opt.idxRef[tp].valid:size(1)
            opt.idxRef[tp].train = torch.zeros(opt.idxRef[tp].train:size(1) - opt.nValidImgs)
            -- Loop through to get proper index values
            validCount = 1
            trainCount = 1
            for i = 1,annot_2d.index:size(1) do
                if validCount <= tmpValid:size(1) and tmpAnnot[i]:equal(tmpValid[validCount]) then
                    opt.idxRef[tp].valid[validCount] = i
                    validCount = validCount + 1
                elseif annot_2d.istrain[i] == 1 then
                    opt.idxRef[tp].train[trainCount] = i
                    trainCount = trainCount + 1
                end
            end

            opt.idxRef2D = opt.idxRef[tp]
            torch.save(opt.save .. '/options.t7', opt)
        end

        self.dataset[tp].annot = annot_2d
        self.dataset[tp].nsamples = {train=opt.idxRef[tp].train:numel(),
                         valid=opt.idxRef[tp].valid:numel(),
                         test=opt.idxRef[tp].test:numel()}
        opt.testIters[tp] = self.dataset[tp].nsamples.test
        opt.validIters[tp] = self.dataset[tp].nsamples.valid


        tp = 'mpi_inf_3dhp'
        if opt.valid3DHP:
            annot_mpi3d = {}
            tags_mpi3d = {'TSId','activity_annotation','data_id','id','univ_annot3','valid_frame', 'bbox', 'annot_2d'}
            a_mpi3d = hdf5.open(paths.concat(projectDir,'data/mpi-inf-3dhp/annotTest.h5'),'r')
            for _,tag in ipairs(tags_mpi3d) do annot_mpi3d[tag] = a_mpi3d:read(tag):all() end
            a_mpi3d:close()

            if not opt.idxRefMPI3D then
                allIdxs = torch.range(1,annot_mpi3d.id:size(1))
                self.dataset[tp].allSamples = annot_mpi3d.id:size(1)
                opt.idxRef[tp] = {}
                opt.idxRef[tp].train = torch.range(1,1)
                opt.idxRef[tp].test = allIdxs
                opt.idxRef[tp].valid = opt.idxRef[tp].test

                opt.idxRefMPI3D = opt.idxRef[tp]
                torch.save(opt.save .. '/options.t7', opt)
            end

            self.dataset[tp].annot = annot_mpi3d
            self.dataset[tp].nsamples = {train = 1,
                             valid=opt.idxRef[tp].valid:numel(),
                             test=opt.idxRef[tp].test:numel()}
            opt.testIters[tp] = self.dataset[tp].nsamples.test
            opt.validIters[tp] = self.dataset[tp].nsamples.valid
        else
            opt.idxRef[tp].train = torch.range(1,1)
            opt.idxRef[tp].valid = torch.range(1,1)
            self.dataset[tp].annot = 1
            self.dataset[tp].nsamples = {train = 1,
                             valid=1,
                             test=1}
            opt.testIters[tp] = 1
            opt.validIters[tp] = 1
        end

        opt.validIters['demo'] = 1



    end

    function Dataset:size(set)
        return {mpii = self.dataset['mpii'].nsamples[set], h36m = self.dataset['h36m'].nsamples[set], mpi_inf_3dhp = self.dataset['mpi_inf_3dhp'].nsamples[set]}
    end

    function Dataset:getPath(idx, tp)
        if tp == 'h36m' then
            folder = string.format('s_%02d_act_%02d_subact_%02d_ca_%02d', self.dataset[tp].annot.subject[idx],
                self.dataset[tp].annot.action[idx], self.dataset[tp].annot.subaction[idx], self.dataset[tp].annot.camera[idx])
            file_name = string.format('%s/%s_%06d.jpg', folder, folder, self.dataset[tp].annot.id[idx])
            return paths.concat(opt.h36mImgDir, file_name)
        elseif tp == 'mpii' then
            return paths.concat(opt.mpiiImgDir,ffi.string(self.dataset[tp].annot.imgname[idx]:char():data()))
        elseif tp == 'mpi_inf_3dhp' then
            file_name = string.format('%d_%d.png', self.dataset[tp].annot.TSId[idx], self.dataset[tp].annot.data_id[idx])
            return paths.concat(opt.mpi_inf_3dhpImgDir, file_name)
        end
    end

    def loadImage(idx, tp):
        if tp == 'demo':
            return image.load(opt.demo)
        else:
            return image.load(self:getPath(idx, tp))

    def Dataset:getPartInfo(idx, tp):
        if tp == 'h36m':
            pts = self.dataset[tp].annot.joint_2d[idx]:clone()
            pts_3d = self.dataset[tp].annot.joint_3d_mono[idx]:clone()
            c = torch.ones(2) * self.h36mImgSize / 2
            s = 1

            root = 8
            for k = 1, 3:
                pts_3d[{{}, k}] = pts_3d[{{}, k}] - pts_3d[{root, k}]
            end

            s2d = 0
            s3d = 0
            for j = 1, self.nBones:
                s2d = s2d +  ((pts[self.skeletonRef[j][1]][1] - pts[self.skeletonRef[j][2]][1]) ^ 2 +
                              (pts[self.skeletonRef[j][1]][2] - pts[self.skeletonRef[j][2]][2]) ^ 2) ^ 0.5
                s3d = s3d +  ((pts_3d[self.skeletonRef[j][1]][1] - pts_3d[self.skeletonRef[j][2]][1]) ^ 2 +
                              (pts_3d[self.skeletonRef[j][1]][2] - pts_3d[self.skeletonRef[j][2]][2]) ^ 2) ^ 0.5

            scale = s2d / s3d

            for j = 1, self.nJoints:
              pts_3d[j][1] = pts_3d[j][1] * scale + pts[root][1]
              pts_3d[j][2] = pts_3d[j][2] * scale + pts[root][2]
              pts_3d[j][3] = pts_3d[j][3] * scale + self.h36mImgSize / 2
            return pts, c, s, pts_3d
        else if tp == 'mpii':
            pts = self.dataset[tp].annot.part[idx]:clone()
            c = self.dataset[tp].annot.center[idx]:clone()
            s = self.dataset[tp].annot.scale[idx]
            -- Small adjustment so cropping is less likely to take feet out
            c[2] = c[2] + 15 * s
            s = s * 1.25
            return pts, c, s
        else if tp == 'mpi_inf_3dhp':
            c = torch.ones(2) * self.mpiinf3dhpImgSize / 2
            s = 1
            pts_3d_ = self.dataset[tp].annot.univ_annot3[idx]:clone()
            pts_2d_ = self.dataset[tp].annot.annot_2d[idx]:clone()
            pts_3d = torch.zeros(self.nJoints, 3)
            annot_2d = torch.zeros(self.nJoints, 2)
            for j = 1, self.nJoints do
                pts_3d[j] = pts_3d_[self.mpiinf3dhgIdx[j]]
                annot_2d[j] = pts_2d_[self.mpiinf3dhgIdx[j]]
            end
            act = self.dataset[tp].annot.activity_annotation[idx]
            bbox = self.dataset[tp].annot.bbox[idx]

            -- print('bbox', bbox)
            scale_2d = math.max(bbox[3], bbox[4]) / 64.
            c1 = (bbox[1] + bbox[3] / 2)
            c2 = (bbox[2] + bbox[4] / 2)
            for j = 1, self.nJoints do
                annot_2d[j][1] = (annot_2d[j][1] - c1) / scale_2d + 32
                annot_2d[j][2] = (annot_2d[j][2] - c2) / scale_2d + 32
            end
            return c, s, pts_3d, act, bbox, annot_2d
        end
    end

    def getPartDetail(self, idx):
        pts = self.dataset['h36m'].annot.joint_2d[idx]:clone()
        pts_3d = self.dataset['h36m'].annot.joint_3d_mono[idx]:clone()
        c = torch.ones(2) * self.h36mImgSize / 2
        s = 1

        s2d = 0
        s3d = 0
        for j = 1, self.nBones do
            s2d = s2d +  ((pts[self.skeletonRef[j][1]][1] - pts[self.skeletonRef[j][2]][1]) ^ 2 +
                          (pts[self.skeletonRef[j][1]][2] - pts[self.skeletonRef[j][2]][2]) ^ 2) ^ 0.5
            s3d = s3d +  ((pts_3d[self.skeletonRef[j][1]][1] - pts_3d[self.skeletonRef[j][2]][1]) ^ 2 +
                          (pts_3d[self.skeletonRef[j][1]][2] - pts_3d[self.skeletonRef[j][2]][2]) ^ 2) ^ 0.5
        end

        root = 8
        scale = s2d / s3d
        c2d = torch.ones(3) * self.h36mImgSize / 2
        c2d[1] = pts[root][1]
        c2d[2] = pts[root][2]


        act = self.dataset['h36m'].annot.action[idx]
        return pts_3d, scale, c2d, pts_3d[root], c, s, act
    end

    return M.Dataset
