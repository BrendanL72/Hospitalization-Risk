#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "svm.h"
#include "eval.h"

typedef std::vector<double> dvec_t; //----OCSVM
typedef std::vector<int>    ivec_t; //----OCSVM

int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

struct svm_node *x;
int max_nr_attr = 64;

struct svm_model* model;
int predict_probability=0;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

//----zyque
/*int compare_1(const void *pa, const void *pb)
{
	double *a = (double *)pa;
	double *b = (double *)pb;
	if(a[1]<b[1]) return -1;
	if(a[1]>b[1]) return +1;
	return 0;
}

int compare_2(const void *pa, const void *pb)
{
	double *a = (double *)pa;
	double *b = (double *)pb;
	if(a[0]<b[0]) return -1;
	if(a[0]>b[0]) return +1;
	return 0;
}


void probability_in_percent(double *prob_values, double *predict_labels, double *true_labels, int len)
{
	double tmp_arr[len][3];
	int index;
	int neg_counter = 0;
	double log_uni = 0, log_pro = 0;
	printf("len: %d\n",(int)len);
	for(index=0;index<len;index++)
	{
		tmp_arr[index][0] = index;
		tmp_arr[index][1] = prob_values[index];
		if(predict_labels[index] == -1) neg_counter+=1;
	}
	qsort(tmp_arr,len,sizeof(tmp_arr[0]),compare_1);
	for(index=0;index<len;index++)
	{
		//in uniform
		tmp_arr[index][1] = (double)(index+1)/len;
		//in proportion
		if(index<neg_counter)
			tmp_arr[index][2] = 0.5*(double)(index+1)/neg_counter;
		else
			tmp_arr[index][2] = 0.5+0.5*(double)(index-neg_counter+1)/(len-neg_counter);
	}
	qsort(tmp_arr,len,sizeof(tmp_arr[0]),compare_2);
	//printf("======================================\n");
	for(index=0;index<len;index++)
	{
		//prob_values[index] = tmp_arr[index][1];
		//printf("index: %d\tvalue: %f\n",(int)tmp_arr[index][0],tmp_arr[index][1]);
		if(true_labels[index] == 1)
		{
			log_uni += log(tmp_arr[index][1]);
			log_pro += log(tmp_arr[index][2]);
		}
		else
		{
			log_uni += log(1-tmp_arr[index][1]);
			log_pro += log(1-tmp_arr[index][2]);
		}
	}
		info("Log likelihood = %g (uniform percentile)\n",log_uni/(double)len);
		info("Log likelihood = %g (segmented percentile)\n",log_pro/(double)len);

}*/
//----OCSVM
void get_prob(const svm_model *model, double *dec_value, double *prob_uni, double *prob_den)
{
	//get probability by scales in model
	//binning
	for(int i=0;i<11;i++)
	{
		double tmp = (model->prob_scales_uniform[i] + model->prob_scales_uniform[i+1])/2;
		if(dec_value[0] < tmp)
		{
			prob_uni[0] = i*0.1;
			break;
		}
		prob_uni[0] = 1;
	}
	for(int i=0;i<11;i++)
	{
		double tmp = (model->prob_scales_density[i] + model->prob_scales_density[i+1])/2;
		if(dec_value[0] < tmp)
		{
			prob_den[0] = i*0.1;
			break;
		}
		prob_den[0] = 1;
	}

	//protection to avoid -inf
	if(prob_uni[0] == 0)
		prob_uni[0] = 0.001;
	else if(prob_uni[0] == 1)
		prob_uni[0] = 0.999;
	if(prob_den[0] == 0)
		prob_den[0] = 0.001;
	else if(prob_den[0] == 1)
		prob_den[0] = 0.999;
}

void predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	int tp = 0, tn = 0, fn = 0, fp = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	int neg_total = 0;
	double log_likeli_platt = 0, log_likeli_uni = 0, log_likeli_den = 0;
	double log_likeli_platt_neg = 0, log_likeli_uni_neg = 0, log_likeli_den_neg = 0;
	int size_prob_values = 40;
	double *dec_value = (double *)malloc(sizeof(double));
	double *prob_uni = (double *)malloc(sizeof(double));
	double *prob_den = (double *)malloc(sizeof(double));
	double *dec_values = (double *)malloc(size_prob_values*sizeof(double));
	double *prob_uniform = (double *)malloc(size_prob_values*sizeof(double));
	double *prob_density = (double *)malloc(size_prob_values*sizeof(double));
	double *prob_values = (double *)malloc(size_prob_values*sizeof(double));
	double *true_labels = (double *)malloc(size_prob_values*sizeof(double));
	double *predict_labels = (double *)malloc(size_prob_values*sizeof(double));
	double *predict_labels_dec = (double *)malloc(size_prob_values*sizeof(double));
	dvec_t dv;	//probability estimates values, only used in eval
	dvec_t dv_dec;	//probability decision value, only used in eval
	ivec_t ty;	//true labels, only used in eval
	dvec_t pl;	//predict lables, only use in eval
	dvec_t pl_dec;	//predict lables(by decision values), only use in eval

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			int *labels=(int *) malloc(nr_class*sizeof(int));
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");
			if(svm_type==ONE_CLASS && predict_probability)
			{
				fprintf(output,"\tPlatt\t\tdec\tuni\tden\ttrue");
			}
			else
			{
				for(j=0;j<nr_class;j++)
					fprintf(output," %d",labels[j]);
			}
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label, predict_label_dec;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		while(1)
		{
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		x[i].index = -1;

		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			predict_label = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else if (predict_probability && svm_type==ONE_CLASS) //----OCSVM
		{
			//probability in platt's
			predict_label = svm_predict_probability(model,x,prob_estimates);	//get predict label and prob_estimates
			//probability in percentile
			predict_label_dec = svm_predict_values(model,x,dec_value);	//get dec_value
			get_prob(model,dec_value,prob_uni,prob_den);
			
			//write into output file
			fprintf(output,"%g",predict_label);
			fprintf(output,"\t%g",prob_estimates[0]);	//Platt scaling
			fprintf(output,"\t%g",dec_value[0]);	//print dec value
			fprintf(output,"\t%g",prob_uni[0]);
			fprintf(output,"\t%g",prob_den[0]);
			fprintf(output,"\t%g",target_label);
			fprintf(output,"\n");

			if(total == size_prob_values)
			{
				//need to resize
				size_prob_values *= 2;
				dec_values = (double *)realloc(dec_values, size_prob_values*sizeof(double));
				prob_values = (double *)realloc(prob_values, size_prob_values*sizeof(double));
				prob_uniform = (double *)realloc(prob_uniform, size_prob_values*sizeof(double));
				prob_density = (double *)realloc(prob_density, size_prob_values*sizeof(double));
				true_labels = (double *)realloc(true_labels, size_prob_values*sizeof(double));
				predict_labels = (double *)realloc(predict_labels, size_prob_values*sizeof(double));
				predict_labels_dec = (double *)realloc(predict_labels_dec, size_prob_values*sizeof(double));
			}
			dec_values[total] = dec_value[0];
			prob_values[total] = prob_estimates[0];	//for platt's method
			prob_uniform[total] = prob_uni[0];
			prob_density[total] = prob_den[0];
			true_labels[total] = target_label;
			predict_labels[total] = predict_label;
			predict_labels_dec[total] = predict_label_dec;
			if(target_label == 1)
			{
				log_likeli_platt += log(prob_values[total]);
				log_likeli_uni += log(prob_uniform[total]);
				log_likeli_den += log(prob_density[total]);
			}
			else
			{
				log_likeli_platt += log(1-prob_values[total]);
				log_likeli_uni += log(1-prob_uniform[total]);
				log_likeli_den += log(1-prob_density[total]);
				log_likeli_platt_neg += log(1-prob_values[total]);
				log_likeli_uni_neg += log(1-prob_uniform[total]);
				log_likeli_den_neg += log(1-prob_density[total]);
				neg_total += 1;
			}
		}
		else
		{
			predict_label = svm_predict(model,x);
			fprintf(output,"%.17g",predict_label);
			double *dec_value = (double *)malloc(sizeof(double));
			svm_predict_values(model,x,dec_value);
			fprintf(output," %g\n",dec_value[0]);
		}

		if(predict_label == target_label)
			++correct;
		if(predict_label == target_label)
		{
			if(target_label > 0)	tp++;
			else	tn++;
		}
		else
		{
			if(target_label > 0)	fn++;
			else	fp++;
		}
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else if (svm_type==ONE_CLASS && predict_probability) //----OCSVM
	{
		dec_values = (double *)realloc(dec_values,total*sizeof(double));
		prob_values = (double *)realloc(prob_values,total*sizeof(double));
		prob_uniform = (double *)realloc(prob_uniform,total*sizeof(double));
		prob_density = (double *)realloc(prob_density,total*sizeof(double));
		true_labels = (double *)realloc(true_labels,total*sizeof(double));
		predict_labels = (double *)realloc(predict_labels,total*sizeof(double));
		predict_labels_dec = (double *)realloc(predict_labels_dec,total*sizeof(double));
		dv.resize(total);
		dv_dec.resize(total);
		ty.resize(total);
		pl.resize(total);
		pl_dec.resize(total);
		//probability_in_percent(prob_values,predict_labels,true_labels,total);
/*		for(int j=0;j<total;j++)
		{
			dv[j] = prob_values[j];
			dv_dec[j] = dec_values[j];
			ty[j] = (int)true_labels[j];
			pl[j] = predict_labels[j];
			pl_dec[j] = predict_labels_dec[j];
		}
		info("===platt's===\n");
		precision(pl,ty);
		recall(pl,ty);
		fscore(pl,ty);
		bac(pl,ty);
		auc(dv,ty);
		ap(dv,ty);
		accuracy(pl,ty);
		info("Log likelihood = %g (probability in Platt)\n",log_likeli_platt/(double)total);
		info("Log likelihood (neg) = %g (probability in Platt)\n",log_likeli_platt_neg/(double)neg_total);
		info("===dec===\n");
		precision(pl_dec,ty);
		recall(pl_dec,ty);
		fscore(pl_dec,ty);
		bac(pl_dec,ty);
		auc(dv_dec,ty);
		ap(dv_dec,ty);
		accuracy(pl_dec,ty);
		info("Log likelihood = %g (probability in uniform)\n",log_likeli_uni/(double)total);
		info("Log likelihood (neg) = %g (probability in uniform)\n",log_likeli_uni_neg/(double)neg_total);
		info("Log likelihood = %g (probability in density)\n",log_likeli_den/(double)total);
		info("Log likelihood (neg) = %g (probability in density)\n",log_likeli_den_neg/(double)neg_total);
		major in outliers, for AUC and AP
		for(int j=0;j<total;j++)
		{
			dv[j] = 1-prob_values[j];
			dv_dec[j] = 1-dec_values[j];
			ty[j] = -(int)true_labels[j];
		}
		info("===platt's(outliers)===\n");
		auc(dv,ty);
		ap(dv,ty);
		info("===dec(outliers)===\n");
		auc(dv_dec,ty);
		ap(dv_dec,ty);
*/
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
/*	info("precision = %g%% (%d/%d) (pos)\n", (double)tp/(tp+fp)*100,tp,tp+fp);
	info("recall = %g%% (%d/%d) (pos)\n", (double)tp/(tp+fn)*100,tp,tp+fn);
	info("precision = %g%% (%d/%d) (neg)\n", (double)tn/(tn+fn)*100,tn,tn+fn);
	info("recall = %g%% (%d/%d) (neg)\n", (double)tn/(tn+fp)*100,tn,tn+fp);
*/
	if(predict_probability)
		free(prob_estimates);
}

void exit_with_help()
{
	printf(
	"Usage: svm-predict [options] test_file model_file output_file\n"
	"options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

int main(int argc, char **argv)
{
	FILE *input, *output;
	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		switch(argv[i-1][1])
		{
			case 'b':
				predict_probability = atoi(argv[i]);
				break;
			case 'q':
				info = &print_null;
				i--;
				break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	if(i>=argc-2)
		exit_with_help();

	input = fopen(argv[i],"r");
	if(input == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[i]);
		exit(1);
	}

	output = fopen(argv[i+2],"w");
	if(output == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",argv[i+2]);
		exit(1);
	}

	if((model=svm_load_model(argv[i+1]))==0)
	{
		fprintf(stderr,"can't open model file %s\n",argv[i+1]);
		exit(1);
	}

	x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
	if(predict_probability)
	{
		if(svm_check_probability_model(model)==0)
		{
			fprintf(stderr,"Model does not support probabiliy estimates\n");
			exit(1);
		}
	}
	else
	{
		if(svm_check_probability_model(model)!=0)
			info("Model supports probability estimates, but disabled in prediction.\n");
	}

	predict(input,output);
	svm_free_and_destroy_model(&model);
	free(x);
	free(line);
	fclose(input);
	fclose(output);
	return 0;
}
