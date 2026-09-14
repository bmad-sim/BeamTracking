function sbs2s(tm::SciBmadStandard; kwargs...)
    radiation_damping_on      = tm.radiation_damping_on
    radiation_fluctuations_on = tm.radiation_fluctuations_on
    ibs_damping_on            = tm.ibs_damping_on
    ibs_fluctuations_on       = tm.ibs_fluctuations_on
    return Symplectic(; radiation_damping_on, radiation_fluctuations_on, ibs_damping_on, ibs_fluctuations_on, kwargs...)
end

# Note: for exact transformations, use Symplectic order 2 because higher orders won't increase accuracy
@inline drift(tm::SciBmadStandard, kc, p_over_q_ref, bunch, L) = drift(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, L) 

@inline pure_rf(tm::SciBmadStandard, kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)                = pure_rf(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, rfparams, beamlineparams, L)                        
@inline pure_bsolenoid(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bm0, L)                              = pure_bsolenoid(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, bm0, L)                             
@inline bsolenoid(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmultipoleparams, L)                      = bsolenoid(sbs2s(tm), kc, p_over_q_ref, bunch, bmultipoleparams, L)                     
@inline pure_bdipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bm1, L)                                = pure_bdipole(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, bm1, L)                               
@inline bdipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmultipoleparams, L)                        = bdipole(sbs2s(tm), kc, p_over_q_ref, bunch, bmultipoleparams, L)                       
@inline pure_bquadrupole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bm2, L)                            = pure_bquadrupole(sbs2s(tm), kc, p_over_q_ref, bunch, bm2, L)                           
@inline bquadrupole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmultipoleparams, L)                    = bquadrupole(sbs2s(tm), kc, p_over_q_ref, bunch, bmultipoleparams, L)                   
@inline pure_bmultipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmk, L)                             = pure_bmultipole(sbs2s(tm), kc, p_over_q_ref, bunch, bmk, L)                            
@inline bmultipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmultipoleparams, L)                     = bmultipole(sbs2s(tm), kc, p_over_q_ref, bunch, bmultipoleparams, L)                    
@inline bmultipole_rf(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L) = bmultipole_rf(sbs2s(tm), kc, p_over_q_ref, bunch, bmultipoleparams, rfparams, beamlineparams, L)
@inline bend_no_field(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, L)                        = bend_no_field(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, bendparams, L)                       
@inline bend_pure_bsolenoid(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bm0, L)             = bend_pure_bsolenoid(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bm0, L)            
@inline bend_bsolenoid(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)     = bend_bsolenoid(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)    
@inline bend_pure_bdipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bm1, L)               = bend_pure_bdipole(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, bendparams, bm1, L)              
@inline bend_bdipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)       = bend_bdipole(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)      
@inline bend_pure_bquadrupole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bm2, L)           = bend_pure_bquadrupole(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bm2, L)          
@inline bend_bquadrupole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)   = bend_bquadrupole(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)  
@inline bend_pure_bmultipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmk, L)            = bend_pure_bmultipole(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmk, L)           
@inline bend_bmultipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)    = bend_bmultipole(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)         
@inline implicit(tm::SciBmadStandard, kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)           = implicit(sbs2s(tm), kc, p_over_q_ref, bunch, bendparams, bmultipoleparams, L)
@inline pure_edipole(tm::SciBmadStandard, kc, p_over_q_ref, bunch, em1, L)                                = pure_edipole(sbs2s(tm; order=2), kc, p_over_q_ref, bunch, em1, L)