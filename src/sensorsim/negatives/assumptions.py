"""Rationales for ALL fixed procedural regions; none are measured populations.

Values are chosen to exercise distinguishable acoustic forms behind a mandatory
realism gate. They cannot justify household representativeness or prevalence.
"""
COMMON = {
    'numerical_support': {'values': {'fs_hz':[8000,96000], 'duration_s':[.1,60],
        'filter_order':2,'lowpass_nyquist_fraction':.8,'band_edges_fs_fraction':[.3,.4],
        'unit_rms_minimum':1e-15},
        'rationale':'Assumption: bounded rendering budget, stable low-order filters and numerical floor; not appliance physics.'},
}
IMPACT = {
    'contact_body': {'values': {'interval_factor':[.65,1.35], 'last_onset_duration_fraction':.68,
        'drop_interval_ratio':.64,'shape':[.7,1.6], 'amplitude':[.65,1.3],
        'drop_amplitude_ratio':.7,'modes':8,'decay_factor':[1,1.8],
        'broadband_weight':1.,'modal_weight':.08,'phase_rad':[0,6.283185307179586]},
        'rationale':'Assumption: jittered, decaying contacts with weak modal ringing avoid identical pulses and pure chimes; hit counts distinguish proxy forms only.'},
}
TONAL = {
    'tonal_forms': {'values': {'base_hz':[650,1000], 'chime_ratios':[1,2.71,5.4],
        'chime_amplitudes':[1,.4,.18],'chime_decay_s':[.35,.65],
        'phone_ratios':[1,1.5],'phone_width_duration':.32,'phone_second_onset_duration':.45,
        'beep_cycles_duty':[3,.42],'timer_cycles_duty':[8,.2],'timer_third_harmonic':.25,
        'jingle_ratios':[1,1.25,1.5,1.125,2],'jingle_second_harmonic':.35,
        'chirp_sweep_hz':3600,'chirp_am_cycles':5,'chirp_am_offset_depth':[.5,.5]},
        'rationale':'Assumption: distinct beep, melody, struck-chime and sweep forms exercise doorbell confusers; all require target-partial exclusion and boundary review.'},
}
NOISE = {
    'device_floor': ({'distribution':'Gaussian unit variance'},'Acoustic test source only; actual device floor is added by the Pi profile.'),
    'room_tone': ({'lowpass_hz':500},'Smooth low-frequency indoor-bed proxy.'),
    'fridge_hum': ({'harmonic_amplitudes':[1,.5,1/3,.25],'noise_weight':.15,'lowpass_hz':700},'Fixed-speed mains compressor proxy; no inverter population coverage.'),
    'distant_traffic': ({'lowpass_hz':[180,350],'envelope_offset_depth':[.4,.6],'envelope_half_cycles':1},'Muffled pass-by envelope, not measured traffic spectra.'),
    'running_tap': ({'band_hz':[300,5000],'am_depth_hz':[.12,1.2]},'Broad splashing proxy with slow modulation.'),
    'boiling_pan': ({'lowpass_hz':1600,'bubble_filter_pole':.92,'bubble_probability':.005,'bubble_weight':3},'Low bubbling bursts on a continuous bed; no measured bubble-rate prior.'),
    'frying': ({'band_hz':[1500,6000],'bed_weight':.25,'crackle_probability':.02,'crackle_fir':[1,-.6]},'Broad high-frequency crackles differentiate frying from bubbling.'),
    'washer_fill': ({'band_hz':[70,650],'am_depth_hz':[.6,.35]},'Slowly modulated low-band fill proxy.'),
    'coffee_machine': ({'band_hz':[700,2500],'pump_hz':5,'pump_offset_depth':[.2,.8],'motor_hz':110,'motor_weight':.3},'Pulsed pump/water proxy; no specific appliance model.'),
}


def parameter_assumptions(family, region, *, impact=False, tonal=False):
    rows = dict(COMMON)
    if region:
        rows['family_region'] = {'values':dict(region), 'rationale':
            'Assumption: disjoint family count/decay/frequency regions make controls distinguishable; not fitted household distributions.'}
    if impact: rows.update(IMPACT)
    if tonal: rows.update(TONAL)
    if family in NOISE:
        values, rationale = NOISE[family]
        rows['noise_form'] = {'values':values, 'rationale':'Assumption: '+rationale}
    if 'motor_hz' in region:
        rows['rotor_form'] = {'values':{'harmonic_decay':'1/k','noise_weight':1},
            'rationale':'Assumption: declining harmonics plus lowpassed broadband rotor noise separate proxy motor forms.'}
    return rows
