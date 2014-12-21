/* Copyright (C) 2005 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.	For further
   information, see the file `LICENSE' included with this distribution. */

package cc.mallet.topics;

import java.util.Arrays;
import java.util.ArrayList;

import java.util.zip.*;

import java.io.*;
import java.text.NumberFormat;

import cc.mallet.types.*;
import cc.mallet.util.Randoms;

/**
 * An implementation of topic model marginal probability estimators
 *  presented in Wallach et al., "Evaluation Methods for Topic Models", ICML (2009)
 * 
 * @author David Mimno
 */

public class MarginalProbEstimator implements Serializable {

	protected int numTopics; // Number of topics to be fit

	// These values are used to encode type/topic counts as
	//  count/topic pairs in a single int.
	protected int topicMask;
	protected int topicBits;

	protected double[] alpha;	 // Dirichlet(alpha,alpha,...) is the distribution over topics
	protected double alphaSum;
	protected double beta;   // Prior on per-topic multinomial distribution over words
	protected double betaSum;
	
	protected double smoothingOnlyMass = 0.0;
	protected double[] cachedCoefficients;

	protected int[][] typeTopicCounts; // indexed by <feature index, topic index>
	protected int[] tokensPerTopic; // indexed by <topic index>

	protected Randoms random;
	
	public MarginalProbEstimator (int numTopics,
								  double[] alpha, double alphaSum,
								  double beta,
								  int[][] typeTopicCounts, 
								  int[] tokensPerTopic) {

		this.numTopics = numTopics;

		if (Integer.bitCount(numTopics) == 1) {
			// exact power of 2
			topicMask = numTopics - 1;
			topicBits = Integer.bitCount(topicMask);
		}
		else {
			// otherwise add an extra bit
			topicMask = Integer.highestOneBit(numTopics) * 2 - 1;
			topicBits = Integer.bitCount(topicMask);
		}

		this.typeTopicCounts = typeTopicCounts;
		this.tokensPerTopic = tokensPerTopic;
		
		this.alphaSum = alphaSum;
		this.alpha = alpha;
		this.beta = beta;
		this.betaSum = beta * typeTopicCounts.length;
		this.random = new Randoms();
		
		cachedCoefficients = new double[ numTopics ];

		// Initialize the smoothing-only sampling bucket
		smoothingOnlyMass = 0;
		
		// Initialize the cached coefficients, using only smoothing.
		//  These values will be selectively replaced in documents with
		//  non-zero counts in particular topics.
		
		for (int topic=0; topic < numTopics; topic++) {
			smoothingOnlyMass += alpha[topic] * beta / (tokensPerTopic[topic] + betaSum);
			cachedCoefficients[topic] =  alpha[topic] / (tokensPerTopic[topic] + betaSum);
		}
		
		System.err.println("Topic Evaluator: " + numTopics + " topics, " + topicBits + " topic bits, " + 
						   Integer.toBinaryString(topicMask) + " topic mask");

	}

	public int[] getTokensPerTopic() { return tokensPerTopic; }
	public int[][] getTypeTopicCounts() { return typeTopicCounts; }

	public double evaluateLeftToRight (InstanceList testing, int numParticles, boolean usingResampling,
									   PrintStream docProbabilityStream) {
		random = new Randoms();

		double logNumParticles = Math.log(numParticles);
		double totalLogLikelihood = 0;
		for (Instance instance : testing) {
			
			FeatureSequence tokenSequence = (FeatureSequence) instance.getData();
      int documentLength = tokenSequence.getLength();
      int predictionIdx = (int)(0.8 * documentLength);

			double docLogLikelihood = 0;

			double[][] particleProbabilities = new double[ numParticles ][];
			for (int particle = 0; particle < numParticles; particle++) {
				particleProbabilities[particle] =
					leftToRight(tokenSequence, usingResampling);
			}

			for (int position = predictionIdx; position < documentLength; position++) {
				double sum = 0;
				for (int particle = 0; particle < numParticles; particle++) {
					sum += particleProbabilities[particle][position];
				}

				if (sum > 0.0) { 
					docLogLikelihood += Math.log(sum) - logNumParticles;
				}
			}

			if (docProbabilityStream != null) {
				docProbabilityStream.println(docLogLikelihood);
			}
			totalLogLikelihood += docLogLikelihood / (documentLength - predictionIdx);
		}

		return totalLogLikelihood / testing.size();
	}
	
	protected double[] leftToRight (FeatureSequence tokenSequence, boolean usingResampling) {

		int[] oneDocTopics = new int[tokenSequence.getLength()];
		double[] wordProbabilities = new double[tokenSequence.getLength()];

		int[] currentTypeTopicCounts;
		int type, oldTopic, newTopic;
		double topicWeightsSum;
		int docLength = tokenSequence.getLength();

		// Keep track of the number of tokens we've examined, not
		//  including out-of-vocabulary words
		int tokensSoFar = 0;

		int[] localTopicCounts = new int[numTopics];
		int[] localTopicIndex = new int[numTopics];

		// Build an array that densely lists the topics that
		//  have non-zero counts.
		int denseIndex = 0;

		// Record the total number of non-zero topics
		int nonZeroTopics = denseIndex;

		//		Initialize the topic count/beta sampling bucket
		double topicBetaMass = 0.0;
		double topicTermMass = 0.0;

		double[] topicTermScores = new double[numTopics];
		int[] topicTermIndices;
		int[] topicTermValues;
		int i;
		double score;

		double logLikelihood = 0;

		// All counts are now zero, we are starting completely fresh.

		//	Iterate over the positions (words) in the document 
		for (int limit = 0; limit < docLength; limit++) {
			
			// Record the marginal probability of the token
			//  at the current limit, summed over all topics.

			if (usingResampling) {

				// Iterate up to the current limit
				for (int position = 0; position < limit; position++) {

					type = tokenSequence.getIndexAtPosition(position);
					oldTopic = oneDocTopics[position];

					// Check for out-of-vocabulary words
					if (type >= typeTopicCounts.length ||
						typeTopicCounts[type] == null) {
						continue;
					}

					currentTypeTopicCounts = typeTopicCounts[type];
				
					//	Remove this token from all counts. 
				
					// Remove this topic's contribution to the 
					//  normalizing constants.
					// Note that we are using clamped estimates of P(w|t),
					//  so we are NOT changing smoothingOnlyMass.
					topicBetaMass -= beta * localTopicCounts[oldTopic] /
						(tokensPerTopic[oldTopic] + betaSum);
				
					// Decrement the local doc/topic counts
				
					localTopicCounts[oldTopic]--;
				
					// Maintain the dense index, if we are deleting
					//  the old topic
					if (localTopicCounts[oldTopic] == 0) {
					
						// First get to the dense location associated with
						//  the old topic.
					
						denseIndex = 0;
					
						// We know it's in there somewhere, so we don't 
						//  need bounds checking.
						while (localTopicIndex[denseIndex] != oldTopic) {
							denseIndex++;
						}
					
						// shift all remaining dense indices to the left.
						while (denseIndex < nonZeroTopics) {
							if (denseIndex < localTopicIndex.length - 1) {
								localTopicIndex[denseIndex] = 
									localTopicIndex[denseIndex + 1];
							}
							denseIndex++;
						}
					
						nonZeroTopics --;
					}

					// Add the old topic's contribution back into the
					//  normalizing constants.
					topicBetaMass += beta * localTopicCounts[oldTopic] /
						(tokensPerTopic[oldTopic] + betaSum);

					// Reset the cached coefficient for this topic
					cachedCoefficients[oldTopic] = 
						(alpha[oldTopic] + localTopicCounts[oldTopic]) /
						(tokensPerTopic[oldTopic] + betaSum);
				

					// Now go over the type/topic counts, calculating the score
					//  for each topic.
				
					int index = 0;
					int currentTopic, currentValue;
				
					boolean alreadyDecremented = false;
				
					topicTermMass = 0.0;
				
					while (index < currentTypeTopicCounts.length && 
						   currentTypeTopicCounts[index] > 0) {
						currentTopic = currentTypeTopicCounts[index] & topicMask;
						currentValue = currentTypeTopicCounts[index] >> topicBits;
					
						score = 
							cachedCoefficients[currentTopic] * currentValue;
						topicTermMass += score;
						topicTermScores[index] = score;
					
						index++;
					}
			
					double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
					double origSample = sample;
				
					//	Make sure it actually gets set
					newTopic = -1;
				
					if (sample < topicTermMass) {
					
						i = -1;
						while (sample > 0) {
							i++;
							sample -= topicTermScores[i];
						}
					
						newTopic = currentTypeTopicCounts[i] & topicMask;
					}
					else {
						sample -= topicTermMass;
					
						if (sample < topicBetaMass) {
							//betaTopicCount++;
						
							sample /= beta;
						
							for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
								int topic = localTopicIndex[denseIndex];
							
								sample -= localTopicCounts[topic] /
									(tokensPerTopic[topic] + betaSum);
							
								if (sample <= 0.0) {
									newTopic = topic;
									break;
								}
							}
						
						}
						else {
							//smoothingOnlyCount++;
						
							sample -= topicBetaMass;
						
							sample /= beta;
						
							newTopic = 0;
							sample -= alpha[newTopic] /
								(tokensPerTopic[newTopic] + betaSum);
						
							while (sample > 0.0) {
								newTopic++;
								sample -= alpha[newTopic] / 
									(tokensPerTopic[newTopic] + betaSum);
							}
						
						}
					
					}
				
					if (newTopic == -1) {
						System.err.println("sampling error: "+ origSample + " " + sample + " " + smoothingOnlyMass + " " + 
										   topicBetaMass + " " + topicTermMass);
						newTopic = numTopics-1; // TODO is this appropriate
						//throw new IllegalStateException ("WorkerRunnable: New topic not sampled.");
					}
					//assert(newTopic != -1);
				
					//			Put that new topic into the counts
					oneDocTopics[position] = newTopic;
				
					topicBetaMass -= beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
					localTopicCounts[newTopic]++;
				
					// If this is a new topic for this document,
					//  add the topic to the dense index.
					if (localTopicCounts[newTopic] == 1) {
					
						// First find the point where we 
						//  should insert the new topic by going to
						//  the end (which is the only reason we're keeping
						//  track of the number of non-zero
						//  topics) and working backwards
					
						denseIndex = nonZeroTopics;
					
						while (denseIndex > 0 &&
							   localTopicIndex[denseIndex - 1] > newTopic) {
						
							localTopicIndex[denseIndex] =
								localTopicIndex[denseIndex - 1];
							denseIndex--;
						}
					
						localTopicIndex[denseIndex] = newTopic;
						nonZeroTopics++;
					}
				
					//	update the coefficients for the non-zero topics
					cachedCoefficients[newTopic] =
						(alpha[newTopic] + localTopicCounts[newTopic]) /
						(tokensPerTopic[newTopic] + betaSum);
				
					topicBetaMass += beta * localTopicCounts[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
				
				}
			}
			
			// We've just resampled all tokens UP TO the current limit,
			//  now sample the token AT the current limit.
			
			type = tokenSequence.getIndexAtPosition(limit);				

			// Check for out-of-vocabulary words
			if (type >= typeTopicCounts.length ||
				typeTopicCounts[type] == null) {
				continue;
			}

			currentTypeTopicCounts = typeTopicCounts[type];

			int index = 0;
			int currentTopic, currentValue;
			
			topicTermMass = 0.0;
			
			while (index < currentTypeTopicCounts.length && 
				   currentTypeTopicCounts[index] > 0) {
				currentTopic = currentTypeTopicCounts[index] & topicMask;
				currentValue = currentTypeTopicCounts[index] >> topicBits;
				
				score = 
					cachedCoefficients[currentTopic] * currentValue;
				topicTermMass += score;
				topicTermScores[index] = score;
				
				//System.out.println("  " + currentTopic + " = " + currentValue);

				index++;
			}
			
			/* // Debugging, to make sure we're getting the right probabilities
			   for (int topic = 0; topic < numTopics; topic++) {
			   index = 0;
			   int displayCount = 0;
				
			   while (index < currentTypeTopicCounts.length &&
			   currentTypeTopicCounts[index] > 0) {
			   currentTopic = currentTypeTopicCounts[index] & topicMask;
			   currentValue = currentTypeTopicCounts[index] >> topicBits;

			   if (currentTopic == topic) {
			   displayCount = currentValue;
			   break;
			   }

			   index++;
			   }
				
			   System.out.print(topic + "\t");
			   System.out.print("(" + localTopicCounts[topic] + " + " + alpha[topic] + ") / " +
			   "(" + alphaSum + " + " + tokensSoFar + ") * ");

			   System.out.println("(" + displayCount + " + " + beta + ") / " +
			   "(" + tokensPerTopic[topic] + " + " + betaSum + ") =" + 
			   ((displayCount + beta) / (tokensPerTopic[topic] + betaSum)));


			   }
			*/
			
			double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);
			double origSample = sample;
				
			// Note that we've been absorbing (alphaSum + docLength) into
			//  the normalizing constant. The true marginal probability needs
			//  this term, so we stick it back in.
			wordProbabilities[limit] +=
				(smoothingOnlyMass + topicBetaMass + topicTermMass) /
				(alphaSum + tokensSoFar);

			//System.out.println("normalizer: " + alphaSum + " + " + tokensSoFar);
			tokensSoFar++;

			//	Make sure it actually gets set
			newTopic = -1;
				
			if (sample < topicTermMass) {
					
				i = -1;
				while (sample > 0) {
					i++;
					sample -= topicTermScores[i];
				}
					
				newTopic = currentTypeTopicCounts[i] & topicMask;
			}
			else {
				sample -= topicTermMass;
					
				if (sample < topicBetaMass) {
					//betaTopicCount++;
						
					sample /= beta;
						
					for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
						int topic = localTopicIndex[denseIndex];
							
						sample -= localTopicCounts[topic] /
							(tokensPerTopic[topic] + betaSum);
							
						if (sample <= 0.0) {
							newTopic = topic;
							break;
						}
					}
						
				}
				else {
					//smoothingOnlyCount++;
						
					sample -= topicBetaMass;
						
					sample /= beta;
						
					newTopic = 0;
					sample -= alpha[newTopic] /
						(tokensPerTopic[newTopic] + betaSum);
						
					while (sample > 0.0) {
						newTopic++;
						sample -= alpha[newTopic] / 
							(tokensPerTopic[newTopic] + betaSum);
					}
						
				}
					
			}
				
			if (newTopic == -1) {
				System.err.println("sampling error: "+ origSample + " " + 
								   sample + " " + smoothingOnlyMass + " " + 
								   topicBetaMass + " " + topicTermMass);
				newTopic = numTopics-1; // TODO is this appropriate
			}
				
			// Put that new topic into the counts
			oneDocTopics[limit] = newTopic;
				
			topicBetaMass -= beta * localTopicCounts[newTopic] /
				(tokensPerTopic[newTopic] + betaSum);
				
			localTopicCounts[newTopic]++;
				
			// If this is a new topic for this document,
			//  add the topic to the dense index.
			if (localTopicCounts[newTopic] == 1) {
					
				// First find the point where we 
				//  should insert the new topic by going to
				//  the end (which is the only reason we're keeping
				//  track of the number of non-zero
				//  topics) and working backwards
					
				denseIndex = nonZeroTopics;
					
				while (denseIndex > 0 &&
					   localTopicIndex[denseIndex - 1] > newTopic) {
						
					localTopicIndex[denseIndex] =
						localTopicIndex[denseIndex - 1];
					denseIndex--;
				}
					
				localTopicIndex[denseIndex] = newTopic;
				nonZeroTopics++;
			}
			
			//	update the coefficients for the non-zero topics
			cachedCoefficients[newTopic] =
				(alpha[newTopic] + localTopicCounts[newTopic]) /
				(tokensPerTopic[newTopic] + betaSum);
				
			topicBetaMass += beta * localTopicCounts[newTopic] /
				(tokensPerTopic[newTopic] + betaSum);
			
			//System.out.println(type + "\t" + newTopic + "\t" + logLikelihood);
			
		}

		//	Clean up our mess: reset the coefficients to values with only
		//	smoothing. The next doc will update its own non-zero topics...

		for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
			int topic = localTopicIndex[denseIndex];

			cachedCoefficients[topic] =
				alpha[topic] / (tokensPerTopic[topic] + betaSum);
		}

		return wordProbabilities;

	}

  public double evaluatePrediction(InstanceList testing,
                                   double predictRatio,
                                   int numParticles,
                                   int numIterations,
                                   PrintStream docProbabilityStream) {

    random = new Randoms();
    double logNumParticles = Math.log(numParticles);
    double totalLogLikelihood = 0;

    for (Instance instance : testing) {
      FeatureSequence tokenSequence = (FeatureSequence) instance.getData();
      int documentLength = tokenSequence.getLength();
      int predictionIdx = (int)(predictRatio * documentLength);
      int numPredictionTokens = documentLength - predictionIdx;

      double docLogLikelihood = 0;
      double[][] particleProbabilities = new double[numParticles][];

      for (int particle = 0; particle < numParticles; particle++) {
        particleProbabilities[particle] = predict(tokenSequence, predictionIdx, numIterations);
      }

      for (int position = 0; position < numPredictionTokens; position++) {
        double sum = 0;
        for (int particle = 0; particle < numParticles; particle++) {
          sum += particleProbabilities[particle][position];
        }

        if (sum > 0.0) {
          docLogLikelihood += Math.log(sum) - logNumParticles;
        }
      }

      totalLogLikelihood += docLogLikelihood / numPredictionTokens;
    }
    return totalLogLikelihood / testing.size();
  }

  protected double[] predict(FeatureSequence tokenSequence,
                             int predictionIdx,
                             int numIterations) {

    int docLength = predictionIdx;
    int[] topics = new int[docLength];

    int[] localTopicCounts = new int[numTopics];
    int[] localTopicIndex = new int[numTopics];
    int tokensSoFar = 0;

    int type;
    int[] currentTypeTopicCounts;

    // Initialize all positions to the most common topic
    // for that type.

    for (int position = 0; position < docLength; position++) {
      type = tokenSequence.getIndexAtPosition(position);

      // Ignore out of vocabulary terms
      if (type < typeTopicCounts.length && typeTopicCounts[type].length != 0) {
        currentTypeTopicCounts = typeTopicCounts[type];

        // This value should be a topic such that
        //  no other topic has more tokens of this type
        //  assigned to it. If for some reason there were
        //  no tokens of this type in the training data, it
        //  will default to topic 0, which is no worse than
        //  random initialization.
        topics[position] = currentTypeTopicCounts[0] & topicMask;

        localTopicCounts[topics[position]]++;
      }
    }

    // Build an array that densely lists the topics that
    //  have non-zero counts.
    int denseIndex = 0;
    for (int topic = 0; topic < numTopics; topic++) {
      if (localTopicCounts[topic] != 0) {
        localTopicIndex[denseIndex] = topic;
        denseIndex++;
      }
    }

    // Record the total number of non-zero topics
    int nonZeroTopics = denseIndex;

    //	  Initialize the topic count/beta sampling bucket
    double topicBetaMass = 0.0;

    // Initialize cached coefficients and the topic/beta
    //  normalizing constant.
    for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
      int topic = localTopicIndex[denseIndex];
      int n = localTopicCounts[topic];

      //  initialize the normalization constant for the (B * n_{t|d}) term
      topicBetaMass += beta * n / (tokensPerTopic[topic] + betaSum);

      //  update the coefficients for the non-zero topics
      cachedCoefficients[topic] = (alpha[topic] + n) / (tokensPerTopic[topic] + betaSum);
    }

    double topicTermMass = 0.0;
    double[] topicTermScores = new double[numTopics];
    int i;
    double score;
    int oldTopic, newTopic;

    for (int iteration = 0; iteration < numIterations; iteration++) {
      // reset tokens counter.
      tokensSoFar = 0;

      //  Iterate over the positions (words) in the document
      for (int position = 0; position < docLength; position++) {
        type = tokenSequence.getIndexAtPosition(position);

        // ignore out-of-vocabulary terms
        if (type >= typeTopicCounts.length || typeTopicCounts[type].length == 0) { continue; }

        oldTopic = topics[position];
        currentTypeTopicCounts = typeTopicCounts[type];

        // Prepare to sample by adjusting existing counts.
        // Note that we do not need to change the smoothing-only
        //  mass since the denominator is clamped.
        topicBetaMass -= beta * localTopicCounts[oldTopic] /
            (tokensPerTopic[oldTopic] + betaSum);

        // Decrement the local doc/topic counts
        localTopicCounts[oldTopic]--;
        //assert(localTopicCounts[oldTopic] >= 0);

        // Maintain the dense index, if we are deleting
        //  the old topic
        if (localTopicCounts[oldTopic] == 0) {

          // First get to the dense location associated with
          //  the old topic.
          denseIndex = 0;

          // We know it's in there somewhere, so we don't
          //  need bounds checking.
          while (localTopicIndex[denseIndex] != oldTopic) {
            denseIndex++;
          }

          // shift all remaining dense indices to the left.
          while (denseIndex < nonZeroTopics) {
            if (denseIndex < localTopicIndex.length - 1) {
              localTopicIndex[denseIndex] =
                  localTopicIndex[denseIndex + 1];
            }
            denseIndex++;
          }

          nonZeroTopics--;
        } // finished maintaining local topic index

        topicBetaMass += beta * localTopicCounts[oldTopic] /
            (tokensPerTopic[oldTopic] + betaSum);

        // Reset the cached coefficient for this topic
        cachedCoefficients[oldTopic] =
            (alpha[oldTopic] + localTopicCounts[oldTopic]) /
                (tokensPerTopic[oldTopic] + betaSum);
        if (cachedCoefficients[oldTopic] <= 0) {
          System.out.println("zero or less coefficient: " + oldTopic + " = (" + alpha[oldTopic] + " + " + localTopicCounts[oldTopic] + ") / ( " + tokensPerTopic[oldTopic] + " + " + betaSum + " );");
        }

        int index = 0;
        int currentTopic, currentValue;
        topicTermMass = 0.0;

        while (index < currentTypeTopicCounts.length &&
            currentTypeTopicCounts[index] > 0) {
          currentTopic = currentTypeTopicCounts[index] & topicMask;
          currentValue = currentTypeTopicCounts[index] >> topicBits;

          score = cachedCoefficients[currentTopic] * currentValue;
          topicTermMass += score;
          topicTermScores[index] = score;

          index++;
        }

        double sample = random.nextUniform() * (smoothingOnlyMass + topicBetaMass + topicTermMass);

        //  Make sure it actually gets set
        newTopic = -1;

        if (sample < topicTermMass) {
          //topicTermCount++;

          i = -1;
          while (sample > 0) {
            i++;
            sample -= topicTermScores[i];
          }

          newTopic = currentTypeTopicCounts[i] & topicMask;
        }
        else {
          sample -= topicTermMass;

          if (sample < topicBetaMass) {
            //betaTopicCount++;

            sample /= beta;

            for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
              int topic = localTopicIndex[denseIndex];

              sample -= localTopicCounts[topic] /
                  (tokensPerTopic[topic] + betaSum);

              if (sample <= 0.0) {
                newTopic = topic;
                break;
              }
            }

          }
          else {
            sample -= topicBetaMass;

            sample /= beta;

            newTopic = 0;
            sample -= alpha[newTopic] /
                (tokensPerTopic[newTopic] + betaSum);

            while (sample > 0.0) {
              newTopic++;

              if (newTopic >= numTopics) {
                index = 0;

                while (index < currentTypeTopicCounts.length &&
                    currentTypeTopicCounts[index] > 0) {
                  currentTopic = currentTypeTopicCounts[index] & topicMask;
                  currentValue = currentTypeTopicCounts[index] >> topicBits;

                  System.out.println(currentTopic + "\t" + currentValue + "\t" + topicTermScores[index] +
                      "\t" + cachedCoefficients[currentTopic]);
                  index++;
                }
              }

              sample -= alpha[newTopic] /
                  (tokensPerTopic[newTopic] + betaSum);
            }

          }
        }

        topics[position] = newTopic;
        tokensSoFar++;

        topicBetaMass -= beta * localTopicCounts[newTopic] /
            (tokensPerTopic[newTopic] + betaSum);

        localTopicCounts[newTopic]++;

        // If this is a new topic for this document,
        //  add the topic to the dense index.
        if (localTopicCounts[newTopic] == 1) {

          // First find the point where we
          //  should insert the new topic by going to
          //  the end (which is the only reason we're keeping
          //  track of the number of non-zero
          //  topics) and working backwards
          denseIndex = nonZeroTopics;

          while (denseIndex > 0 &&
              localTopicIndex[denseIndex - 1] > newTopic) {

            localTopicIndex[denseIndex] =
                localTopicIndex[denseIndex - 1];
            denseIndex--;
          }

          localTopicIndex[denseIndex] = newTopic;
          nonZeroTopics++;
        }

        //  update the coefficients for the non-zero topics
        cachedCoefficients[newTopic] =
            (alpha[newTopic] + localTopicCounts[newTopic]) /
                (tokensPerTopic[newTopic] + betaSum);

        topicBetaMass += beta * localTopicCounts[newTopic] /
            (tokensPerTopic[newTopic] + betaSum);

      }
    }

    // start predicting the rest of the document based on current local counts.
    double[] wordProbabilities = new double[tokenSequence.getLength() - predictionIdx];

    for (int position = predictionIdx; position < tokenSequence.size(); position++) {

      type = tokenSequence.getIndexAtPosition(position);

      // Check for out-of-vocabulary words
      if (type >= typeTopicCounts.length ||
          typeTopicCounts[type] == null) {
        continue;
      }

      currentTypeTopicCounts = typeTopicCounts[type];

      int index = 0;
      int currentTopic, currentValue;

      topicTermMass = 0.0;

      while (index < currentTypeTopicCounts.length &&
          currentTypeTopicCounts[index] > 0) {
        currentTopic = currentTypeTopicCounts[index] & topicMask;
        currentValue = currentTypeTopicCounts[index] >> topicBits;

        score = cachedCoefficients[currentTopic] * currentValue;
        topicTermMass += score;
        topicTermScores[index] = score;
        index++;
      }

      // Note that we've been absorbing (alphaSum + docLength) into
      //  the normalizing constant. The true marginal probability needs
      //  this term, so we stick it back in.
      wordProbabilities[position - predictionIdx] =
          (smoothingOnlyMass + topicBetaMass + topicTermMass) / (alphaSum + tokensSoFar);
    }

    //	Clean up our mess: reset the coefficients to values with only
    //	smoothing. The next doc will update its own non-zero topics...
    for (denseIndex = 0; denseIndex < nonZeroTopics; denseIndex++) {
      int topic = localTopicIndex[denseIndex];

      cachedCoefficients[topic] =
          alpha[topic] / (tokensPerTopic[topic] + betaSum);
    }

    return wordProbabilities;
  }

	private static final long serialVersionUID = 1;
    private static final int CURRENT_SERIAL_VERSION = 0;
    private static final int NULL_INTEGER = -1;

    private void writeObject (ObjectOutputStream out) throws IOException {
        out.writeInt (CURRENT_SERIAL_VERSION);

        out.writeInt(numTopics);

        out.writeInt(topicMask);
        out.writeInt(topicBits);

        out.writeObject(alpha);
        out.writeDouble(alphaSum);
        out.writeDouble(beta);
		out.writeDouble(betaSum);

		out.writeObject(typeTopicCounts);
		out.writeObject(tokensPerTopic);

		out.writeObject(random);

        out.writeDouble(smoothingOnlyMass);
		out.writeObject(cachedCoefficients);
    }

	private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {

        int version = in.readInt ();

        numTopics = in.readInt();

        topicMask = in.readInt();
        topicBits = in.readInt();

        alpha = (double[]) in.readObject();
		alphaSum = in.readDouble();
        beta = in.readDouble();
        betaSum = in.readDouble();

        typeTopicCounts = (int[][]) in.readObject();
        tokensPerTopic = (int[]) in.readObject();

        random = (Randoms) in.readObject();

        smoothingOnlyMass = in.readDouble();
        cachedCoefficients = (double[]) in.readObject();
    }

    public static MarginalProbEstimator read (File f) throws Exception {

        MarginalProbEstimator estimator = null;

        ObjectInputStream ois = new ObjectInputStream (new FileInputStream(f));
        estimator = (MarginalProbEstimator) ois.readObject();
        ois.close();

        return estimator;
    }


}
