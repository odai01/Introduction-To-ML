#################################
# Your name: Odai Agbaria, Id: 212609440
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals

class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        samples = np.zeros((m, 2)) 
        for i in range(m):
            x = np.random.uniform(0, 1) 
            if (0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1):
                p_y_given_x = 0.8
            else:
                p_y_given_x = 0.1
            y = np.random.binomial(1, p_y_given_x)  # Draw y based on P[y=1|x]
            samples[i] = [x, y]
        return samples


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        trials=int(((m_last-m_first)/step)+1)
        I=[(0,0.2),(0.4,0.6),(0.8,1)]
        errors_array = np.zeros((trials, 2))
        emp_errors=np.zeros(trials)
        true_errors=np.zeros(trials)
        index=0
        n_values = np.arange(m_first, m_last+1, step)
        k=3
        for i in n_values:
            temp_emp_error=0
            temp_true_error=0
            for j in range(T):
                samples=self.sample_from_D(i)
                sorted_indices = np.argsort(samples[:, 0])
                samples=samples[sorted_indices]
                Xvalues=samples[:,0]
                Ylabels=samples[:,1]
                Intervals,Emp_Error=intervals.find_best_interval(Xvalues,Ylabels,k)
                temp_emp_error+= (Emp_Error/i)
                temp_true_error+=self.calculate_true_error(Intervals,I)

            emp_errors[index]=(temp_emp_error/T)
            true_errors[index]=(temp_true_error/T)
            errors_array[index]=(emp_errors[index],true_errors[index])
            index+=1

        plt.plot(n_values, true_errors, label='Average True Error', color='black',marker='x')
        plt.plot(n_values, emp_errors, label='Average Empirical Error', color='purple',marker='o')
        plt.xlabel('Sample Size (n)')
        plt.ylabel('Error')
        plt.title('Empirical vs. True Error as a Function of Sample Size')
        plt.legend()
        plt.show()
        return errors_array
    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        trials=k_last-k_first+1
        I=[(0,0.2),(0.4,0.6),(0.8,1)]
        n=1500
        samples=self.sample_from_D(n)
        sorted_indices = np.argsort(samples[:, 0])
        samples=samples[sorted_indices]
        Xvalues=samples[:,0]
        Ylabels=samples[:,1]
        emp_errors=np.zeros(trials)
        true_errors=np.zeros(trials)
        index=0
        k_values = np.arange(k_first, k_last+1, step)
        for k in k_values:
            Intervals,Emp_Error=intervals.find_best_interval(Xvalues,Ylabels,k)
            emp_errors[index]=(Emp_Error/n)
            true_errors[index]=self.calculate_true_error(Intervals,I)
            if(index==0):
                min_emp_error=emp_errors[0]
                bestk=k
            else:
                if(emp_errors[index]<min_emp_error):
                    min_emp_error=emp_errors[index]
                    bestk=k
            index+=1

        plt.plot(k_values, true_errors, label='Average True Error', color='red',marker='*')
        plt.plot(k_values, emp_errors, label='Average Empirical Error', color='green',marker='*')
        plt.xlabel('K')
        plt.ylabel('Error')
        plt.title('Empirical vs. True Error as a Function of K')
        plt.legend()
        plt.show()
        # note: i am returning the best k with the minimal empirical errorn, not true error
        # in the forum it was said that it doesnt matter which one we return.
        return bestk

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        trials=k_last-k_first+1
        I=[(0,0.2),(0.4,0.6),(0.8,1)]
        n=1500
        samples=self.sample_from_D(n)
        sorted_indices = np.argsort(samples[:, 0])
        samples=samples[sorted_indices]
        Xvalues=samples[:,0]
        Ylabels=samples[:,1]
        emp_errors=np.zeros(trials)
        true_errors=np.zeros(trials)
        penalties=np.zeros(trials)
        sum_penalties_emp=np.zeros(trials)
        delta=0.1
        index=0
        k_values = np.arange(k_first, k_last+1, step)
        for k in k_values:
            Intervals,Emp_Error=intervals.find_best_interval(Xvalues,Ylabels,k)
            emp_errors[index]=(Emp_Error/n)
            true_errors[index]=self.calculate_true_error(Intervals,I)
            Vcdim=2*k
            penalties[index]=2*np.sqrt((Vcdim+np.log(2/delta))/n)
            sum_penalties_emp[index]=penalties[index]+emp_errors[index]
            if(index==0):
                min_err=sum_penalties_emp[0]
                bestk=k
            else:
                if(sum_penalties_emp[index]<min_err):
                    min_err=sum_penalties_emp[index]
                    bestk=k
            index+=1

        plt.plot(k_values, true_errors, label='True Error', color='red',marker='*')
        plt.plot(k_values, emp_errors, label='Empirical Error', color='green',marker='*')
        plt.plot(k_values, penalties, label='Penalty', color='blue',marker='*')
        plt.plot(k_values, sum_penalties_emp, label='Sum of Empirical Error and Penalty', color='yellow',marker='*')
        plt.xlabel('K')
        plt.ylabel('Error\Penalty')
        plt.title('Qustion 2d')
        plt.legend()
        plt.show()
        return bestk

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        n=m
        n1=int(0.8*n)
        n2=int(0.2*n)
        samples=self.sample_from_D(n)
        training_samples=samples[:n1,:]
        sorted_indices = np.argsort(training_samples[:, 0])
        training_samples=training_samples[sorted_indices]
        holdout_samples=samples[n1:,:]
        sorted_indices = np.argsort(holdout_samples[:, 0])
        holdout_samples=holdout_samples[sorted_indices]
        X_training_values=training_samples[:,0]
        Y_training_labels=training_samples[:,1]
        X_holdout_values=holdout_samples[:,0]
        Y_holdout_labels=holdout_samples[:,1]
        k_values = np.arange(1, 11)
        H=[]
        emp_erros_on_holdut=np.zeros(10)
        for k in k_values:
            Intervals,Emp_Error=intervals.find_best_interval(X_training_values,Y_training_labels,k)
            H.append(Intervals)
            err_count=self.calculate_emp_error(X_holdout_values,Y_holdout_labels,Intervals)
            emp_erros_on_holdut[k-1]=(err_count/n2)
        
        for i in range(len(H)):
            if(i==0):
                min_emp_error=emp_erros_on_holdut[0]
                bestk=0
            if(emp_erros_on_holdut[i]<min_emp_error):
                min_emp_error=emp_erros_on_holdut[i]
                bestk=i+1
        plt.plot(k_values, emp_erros_on_holdut, label='Empirical Error on the holdut set', color='red',marker='*')
        plt.xlabel('K')
        plt.ylabel('Empirical Error')
        plt.title('Question (e)')
        plt.legend()
        plt.show()
        return bestk
    #################################
    # Place for additional methods
    def Intersection_Intervals_length(self,intervals1,intervals2):
        intersections_length=0
        for i in intervals1:
            for j in intervals2:
                intersection_start=max(i[0],j[0])
                intersection_end=min(i[1],j[1])
                if(intersection_start<intersection_end):
                    intersections_length+=(intersection_end-intersection_start)
        return intersections_length
    
    def complementary_intervals(self,intervals):
        complementary_intervals = []
        if intervals[0][0] > 0:
            complementary_intervals.append((0, intervals[0][0]))
        
        for i in range(1, len(intervals)):
            prev_end = intervals[i-1][1]
            current_start = intervals[i][0]
            if prev_end < current_start:
                complementary_intervals.append((prev_end, current_start))
        
        if intervals[-1][1] < 1:
            complementary_intervals.append((intervals[-1][1], 1))
        return complementary_intervals
    
    def calculate_true_error(self,hypothesis_intervals,true_intervals):
        res=0
        comp_hypothesis_intervals=self.complementary_intervals(hypothesis_intervals)
        comp_true_intervals=self.complementary_intervals(true_intervals)
        res+= 0.2*self.Intersection_Intervals_length(hypothesis_intervals,true_intervals)
        res+= 0.9*self.Intersection_Intervals_length(hypothesis_intervals,comp_true_intervals)
        res+= 0.8*self.Intersection_Intervals_length(comp_hypothesis_intervals,true_intervals)
        res+= 0.1*self.Intersection_Intervals_length(comp_hypothesis_intervals,comp_true_intervals)
        return res

    def calculate_emp_error(self,xvalues,ylabels,hypothesis_intervals):
        error_count=0
        for i in range(len(xvalues)):
            x=xvalues[i]
            real_y=ylabels[i]
            emp_y=0
            for start,end in hypothesis_intervals:
                if(x>=start and x<=end):
                    emp_y=1
                    break

            if(emp_y!=real_y):
                error_count+=1

        return error_count
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

