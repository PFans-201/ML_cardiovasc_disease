# Milestone M2: Bayesian Network Design

**Deadline:** October 20, 23:59  
**Status:** 📋 Planned

---

## 📋 Objectives

1. **Structure Design**
   - Design Bayesian network connecting risk factors, disease states, and observations
   - Include minimum required variables: risk_score, state, treatment, utility
   - Add subset of symptoms/lab values for evidence representation
   - Justify edge directions with clinical reasoning

2. **Variable Discretization**
   - Convert continuous variables to categorical for simple BN implementation
   - Design appropriate thresholds based on medical knowledge
   - Work with imputed dataset from M1 (not normalized)
   - Document discretization rationale

3. **Parameter Estimation**
   - Estimate Conditional Probability Tables (CPTs) using relative frequencies
   - Treat encounters as independent observations for counting
   - Handle variables with multiple parents appropriately
   - Display network structure and sample CPTs

4. **Independence Analysis**
   - Identify conditional independencies implied by BN structure
   - Select clinically interesting independence relationships
   - Evaluate medical plausibility of independence assumptions

---

## 📊 Specific Tasks

### 7.1. Structure Design
- [ ] Design BN including: risk_score, state, treatment, utility (minimum)
- [ ] Add subset of symptoms and/or lab values as observable evidence
- [ ] Justify each edge with clinical reasoning
- [ ] Implement network structure in pgmpy
- [ ] Visualize network graph with clear node labels

### 7.2. Discretization
- [ ] Select appropriate thresholds for continuous variables
- [ ] Use imputed dataset (not normalized) for discretization
- [ ] Create categorical bins with medical justification
- [ ] Document discretization rules and rationale
- [ ] Validate discretization preserves meaningful patterns

### 7.3. Estimating CPTs by Counting
- [ ] Compute relative frequencies for all CPT entries
- [ ] Handle variables with multiple parents correctly
- [ ] Use full dataset treating encounters as independent
- [ ] Display at least one complete CPT as example
- [ ] Verify CPT probabilities sum to 1.0

### 7.4. Analysis of Independencies
- [ ] Use pgmpy's get_independencies() to list all conditional independencies
- [ ] Select clinically interesting independence relationships
- [ ] Evaluate medical plausibility of independence assumptions
- [ ] Discuss whether independencies seem reasonable for cardiovascular disease

---

## 🎯 Key Requirements

- **Minimum Variables**: Must include risk_score, state, treatment, utility
- **Observable Evidence**: Include symptoms/labs so BN represents how state generates observations
- **Clinical Justification**: Each edge should have medical reasoning
- **Independent Observations**: Treat each encounter row as separate for CPT estimation
- **Discretization**: Use medical knowledge to create meaningful categorical bins

---

## 📁 Deliverables

1. **Jupyter Notebook**: `M2_G14.ipynb` with complete BN design process
2. **Network Visualization**: Clear graph showing structure and relationships
3. **CPT Documentation**: Sample probability tables with interpretation
4. **Independence Analysis**: Discussion of conditional independence implications
5. **Clinical Justification**: Medical reasoning for network structure decisions

---

## 🔗 Connections to Future Milestones

- **M3**: This BN structure will be used for exact and approximate inference
- **M4**: Hand-designed structure will be compared against data-learned structure
- **M5**: BN insights about state relationships will inform HMM design
- **M6**: Understanding of state-treatment-utility relationships will guide RL reward design

---

## 🛠 Technical Notes

- **Discretization Strategy**: Balance between information preservation and BN simplicity
- **CPT Sparsity**: Handle cases where some parent value combinations are rare
- **Network Complexity**: Start simple - can always add more variables later
- **Medical Validity**: Structure should reflect causal relationships, not just correlations
- **pgmpy Implementation**: Use standard pgmpy classes and methods for consistency
