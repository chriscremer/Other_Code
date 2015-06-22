

%data = importdata('/data1/morrislab/ccremer/wrana_data1/ProCoding_all_RPMs_Tx_cleaned_ID.txt');


data = dlmread('/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_metastatic_exps.txt', ' ', 0, 1);

data1 = dlmread('/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_other_exps.txt', ' ', 0, 1);


%data_metastatic = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_metastatic_exps.txt'
%data_other = '/data1/morrislab/ccremer/TCGA_data/find_metastatic/output_other_exps.txt'



%set_1_columns = [];
%set_2_columns = [];

%rememeber that textdata is offset by one because it has GeneID taking the first column
%for n = 2:50

	%if ~isempty(strmatch(data.textdata(1,n), samples_to_analyze))

	%	set_1_columns = cat(1, set_1_columns, n-1);
	%else 

		%set_2_columns = cat(1, set_2_columns, n-1);

	%end
%end



meta_panel = data(:, 1:74);
other_panel = data1(:, 1:74);

meta_test = data(:, 75:148);
other_test = data1(:, 75:148);

normaldata = horzcat(meta_panel, other_panel);
tumordata = horzcat(meta_test, other_test);




%run ISOpure S1

addpath(genpath('/data1/morrislab/ccremer/ISOpureMatlab/ISOpureS1'));

disp('---------------------');
disp('running ISOpure step 1...');

[ISOpureS1model loglikelihood] = learnmodel(tumordata, normaldata);

theta = ISOpureS1model.theta;


file = ['output_files/test_1.mat'];
%file2 = ['output_files/test_1_columns.mat'];


save(file, 'theta');
%save(file2, 'set_1_columns')

rmpath(genpath('ISOpureS1'));





%{

normaldata = [];

for n = set_1_columns

	normaldata = cat(1, normaldata, data.data(:, n));

end

tumordata = [];

for n = set_2_columns

	tumordata = cat(1, tumordata, data.data(:, n));

end



for n = 1:49

	tumordata = data.data(:,n);

	LG_normaldata = [];
	a = 1;
	while a == 1

		LG_normaldata_sample = randi([1,22], 1, 1);

		if ~(LG_normaldata_sample == n) & isempty(find(LG_normaldata==LG_normaldata_sample))

			LG_normaldata = cat(1, LG_normaldata, LG_normaldata_sample);

			s = size(LG_normaldata);

			if s(1) == 6
				a = 2;
			end

		end

	end


	HG_normaldata = [];
	a = 1;
	while a == 1

		HG_normaldata_sample = randi([31,49], 1, 1);

		if ~(HG_normaldata_sample == n) & isempty(find(HG_normaldata==HG_normaldata_sample))

			HG_normaldata = cat(1, HG_normaldata, HG_normaldata_sample);

			s = size(HG_normaldata);

			if s(1) == 6
				a = 2;
			end

		end

	end

	normaldata_indexes = cat(1, LG_normaldata, HG_normaldata);

	normaldata = data.data(:, normaldata_indexes);


	%run ISOpure S1

	addpath(genpath('/data1/morrislab/ccremer/ISOpureMatlab/ISOpureS1'));

	disp('---------------------');
	disp('running ISOpure step 1...');

	[ISOpureS1model loglikelihood] = learnmodel(tumordata, normaldata);

	theta = ISOpureS1model.theta;

	%str = int2str(n);
	file = ['output_files/' str '.mat'];

	save(file, 'theta');

	rmpath(genpath('ISOpureS1'));


end



%}
