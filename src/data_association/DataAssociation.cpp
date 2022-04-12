#include <Eigen/Core>
#include <vector>
#include "data_association/DataAssociation.h"

// Basically all Hungarian algorithm code was taken from https://github.com/mcximing/hungarian-algorithm-cpp


std::ostream& operator<<(std::ostream& os, const da::AssociationMethod& asso_method) {
  switch (asso_method) {
    case da::AssociationMethod::MaximumLikelihood: {
      os << "MaximumLikelihood";
      break;
    }
    case da::AssociationMethod::KnownDataAssociation: {
      os << "KnownDataAssociation";
      break;
    }
  }

  return os;
}


namespace da {

  double chi2inv(double p, unsigned int dim)
  {
    boost::math::chi_squared dist(dim);
    return quantile(dist, p);
  }

  std::vector<int> auction(const Eigen::MatrixXd& problem, double eps, uint64_t max_iterations) {
    int m = problem.rows();
    int n = problem.cols();

    std::cout << "Starting auction with problem size (" << m << ", " << n << ")\n";

    std::deque<int> unassigned_queue;
    std::vector<int> assigned_landmarks;

    // Initilize
    for (int i = 0; i < n; i++) {
      unassigned_queue.push_back(i);
      assigned_landmarks.push_back(-1);
    }

    // Use Eigen vector for convenience below
    Eigen::VectorXd prices(m);
    for (int i = 0; i < m; i++) {
      prices(i) = 0;
    }

    uint64_t curr_iter = 0;

    while (!unassigned_queue.empty() && curr_iter < max_iterations) {
      int l_star = unassigned_queue.front();
      unassigned_queue.pop_front();

      if (curr_iter > max_iterations) {
        break;
      }
      Eigen::MatrixXd::Index i_star;
      double val_max = (problem.col(l_star) - prices).maxCoeff(&i_star);

      auto prev_owner = std::find(assigned_landmarks.begin(), assigned_landmarks.end(), i_star);
      assigned_landmarks[l_star] = i_star;

      if (prev_owner != assigned_landmarks.end()) {
        // The item has a previous owner
        *prev_owner = -1;
        int pos = std::distance(assigned_landmarks.begin(), prev_owner);
        unassigned_queue.push_back(pos);
      }

      double y = problem(i_star, l_star) - val_max;
      prices(i_star) += y + eps;
      curr_iter++;
    }

    if (curr_iter >= max_iterations) {
      std::cout << "\x1B[31m" << "Auction terminated early!\n" << "\033[0m";
    } else {
      std::cout << "\x1B[32m" << "Auction terminated successfully after " << curr_iter << " iterations!\n" << "\033[0m";      
    }

    std::cout << "Solution from auction:\n";
    for (int i = 0; i < assigned_landmarks.size(); i++) {
      std::cout << "Landmark " << i << " with measurement " << assigned_landmarks[i] << "\n";
    }

    return assigned_landmarks;
  }

using namespace std;

void assignmentoptimal(int *assignment, double *cost, const double *distMatrix, int nOfRows, int nOfColumns);
void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);
void computeassignmentcost(int *assignment, double *cost, const double *distMatrix, int nOfRows);
void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);


std::vector<int> hungarian(const Eigen::MatrixXd &cost_matrix)
{
	unsigned int nRows = cost_matrix.rows();
	unsigned int nCols = cost_matrix.cols();

	std::vector<int> assignment;
	assignment.resize(nRows);
	double cost = 0;

	// call solving function
	assignmentoptimal(assignment.data(), &cost, cost_matrix.data(), nRows, nCols);

	return assignment;
}

//********************************************************//
// Solve optimal solution for assignment problem using Munkres algorithm, also known as Hungarian Algorithm.
//********************************************************//
void assignmentoptimal(int *assignment, double *cost, const double * distMatrixIn, int nOfRows, int nOfColumns)
{
	double *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
	bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
	int nOfElements, minDim, row, col;

	/* initialization */
	*cost = 0;
	for (row = 0; row < nOfRows; row++)
		assignment[row] = -1;

	/* generate working copy of distance Matrix */
	/* check if all matrix elements are positive */
	nOfElements = nOfRows * nOfColumns;
	distMatrix = (double *)malloc(nOfElements * sizeof(double));
	distMatrixEnd = distMatrix + nOfElements;

	for (row = 0; row < nOfElements; row++)
	{
		value = distMatrixIn[row];
		if (value < 0)
			cerr << "All matrix elements have to be non-negative." << endl;
		distMatrix[row] = value;
	}

	/* memory allocation */
	coveredColumns = (bool *)calloc(nOfColumns, sizeof(bool));
	coveredRows = (bool *)calloc(nOfRows, sizeof(bool));
	starMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	primeMatrix = (bool *)calloc(nOfElements, sizeof(bool));
	newStarMatrix = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4 */

	/* preliminary steps */
	if (nOfRows <= nOfColumns)
	{
		minDim = nOfRows;

		for (row = 0; row < nOfRows; row++)
		{
			/* find the smallest element in the row */
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while (distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if (value < minValue)
					minValue = value;
				distMatrixTemp += nOfRows;
			}

			/* subtract the smallest element from each element of the row */
			distMatrixTemp = distMatrix + row;
			while (distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}

		/* Steps 1 and 2a */
		for (row = 0; row < nOfRows; row++)
			for (col = 0; col < nOfColumns; col++)
				if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON)
					if (!coveredColumns[col])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						break;
					}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		minDim = nOfColumns;

		for (col = 0; col < nOfColumns; col++)
		{
			/* find the smallest element in the column */
			distMatrixTemp = distMatrix + nOfRows * col;
			columnEnd = distMatrixTemp + nOfRows;

			minValue = *distMatrixTemp++;
			while (distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if (value < minValue)
					minValue = value;
			}

			/* subtract the smallest element from each element of the column */
			distMatrixTemp = distMatrix + nOfRows * col;
			while (distMatrixTemp < columnEnd)
				*distMatrixTemp++ -= minValue;
		}

		/* Steps 1 and 2a */
		for (col = 0; col < nOfColumns; col++)
			for (row = 0; row < nOfRows; row++)
				if (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON)
					if (!coveredRows[row])
					{
						starMatrix[row + nOfRows * col] = true;
						coveredColumns[col] = true;
						coveredRows[row] = true;
						break;
					}
		for (row = 0; row < nOfRows; row++)
			coveredRows[row] = false;
	}

	/* move to step 2b */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, distMatrixIn, nOfRows);

	/* free allocated memory */
	free(distMatrix);
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newStarMatrix);

	return;
}

/********************************************************/
void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns)
{
	int row, col;

	for (row = 0; row < nOfRows; row++)
		for (col = 0; col < nOfColumns; col++)
			if (starMatrix[row + nOfRows * col])
			{
#ifdef ONE_INDEXING
				assignment[row] = col + 1; /* MATLAB-Indexing */
#else
				assignment[row] = col;
#endif
				break;
			}
}

/********************************************************/
void computeassignmentcost(int *assignment, double *cost, const double *distMatrix, int nOfRows)
{
	int row, col;

	for (row = 0; row < nOfRows; row++)
	{
		col = assignment[row];
		if (col >= 0)
			*cost += distMatrix[row + nOfRows * col];
	}
}

/********************************************************/
void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool *starMatrixTemp, *columnEnd;
	int col;

	/* cover every column containing a starred zero */
	for (col = 0; col < nOfColumns; col++)
	{
		starMatrixTemp = starMatrix + nOfRows * col;
		columnEnd = starMatrixTemp + nOfRows;
		while (starMatrixTemp < columnEnd)
		{
			if (*starMatrixTemp++)
			{
				coveredColumns[col] = true;
				break;
			}
		}
	}

	/* move to step 3 */
	step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	int col, nOfCoveredColumns;

	/* count covered columns */
	nOfCoveredColumns = 0;
	for (col = 0; col < nOfColumns; col++)
		if (coveredColumns[col])
			nOfCoveredColumns++;

	if (nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		/* move to step 3 */
		step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}
}

/********************************************************/
void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	bool zerosFound;
	int row, col, starCol;

	zerosFound = true;
	while (zerosFound)
	{
		zerosFound = false;
		for (col = 0; col < nOfColumns; col++)
			if (!coveredColumns[col])
				for (row = 0; row < nOfRows; row++)
					if ((!coveredRows[row]) && (fabs(distMatrix[row + nOfRows * col]) < DBL_EPSILON))
					{
						/* prime zero */
						primeMatrix[row + nOfRows * col] = true;

						/* find starred zero in current row */
						for (starCol = 0; starCol < nOfColumns; starCol++)
							if (starMatrix[row + nOfRows * starCol])
								break;

						if (starCol == nOfColumns) /* no starred zero found */
						{
							/* move to step 4 */
							step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
							return;
						}
						else
						{
							coveredRows[row] = true;
							coveredColumns[starCol] = false;
							zerosFound = true;
							break;
						}
					}
	}

	/* move to step 5 */
	step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col)
{
	int n, starRow, starCol, primeRow, primeCol;
	int nOfElements = nOfRows * nOfColumns;

	/* generate temporary copy of starMatrix */
	for (n = 0; n < nOfElements; n++)
		newStarMatrix[n] = starMatrix[n];

	/* star current zero */
	newStarMatrix[row + nOfRows * col] = true;

	/* find starred zero in current column */
	starCol = col;
	for (starRow = 0; starRow < nOfRows; starRow++)
		if (starMatrix[starRow + nOfRows * starCol])
			break;

	while (starRow < nOfRows)
	{
		/* unstar the starred zero */
		newStarMatrix[starRow + nOfRows * starCol] = false;

		/* find primed zero in current row */
		primeRow = starRow;
		for (primeCol = 0; primeCol < nOfColumns; primeCol++)
			if (primeMatrix[primeRow + nOfRows * primeCol])
				break;

		/* star the primed zero */
		newStarMatrix[primeRow + nOfRows * primeCol] = true;

		/* find starred zero in current column */
		starCol = primeCol;
		for (starRow = 0; starRow < nOfRows; starRow++)
			if (starMatrix[starRow + nOfRows * starCol])
				break;
	}

	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for (n = 0; n < nOfElements; n++)
	{
		primeMatrix[n] = false;
		starMatrix[n] = newStarMatrix[n];
	}
	for (n = 0; n < nOfRows; n++)
		coveredRows[n] = false;

	/* move to step 2a */
	step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim)
{
	double h, value;
	int row, col;

	/* find smallest uncovered element h */
	h = DBL_MAX;
	for (row = 0; row < nOfRows; row++)
		if (!coveredRows[row])
			for (col = 0; col < nOfColumns; col++)
				if (!coveredColumns[col])
				{
					value = distMatrix[row + nOfRows * col];
					if (value < h)
						h = value;
				}

	/* add h to each covered row */
	for (row = 0; row < nOfRows; row++)
		if (coveredRows[row])
			for (col = 0; col < nOfColumns; col++)
				distMatrix[row + nOfRows * col] += h;

	/* subtract h from each uncovered column */
	for (col = 0; col < nOfColumns; col++)
		if (!coveredColumns[col])
			for (row = 0; row < nOfRows; row++)
				distMatrix[row + nOfRows * col] -= h;

	/* move to step 3 */
	step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
} // namespace da
